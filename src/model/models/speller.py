import numpy as np
import torch
from models.attention import MultiHeadAttention
from torch import nn
from torch.autograd import Variable

from model.config import CFG


class Speller(nn.Module):
    """
    Decoder class used to identify the characters associated with each section
    in the voice recording
    """

    def __init__(
        self,
        num_classes,
        hidden_dim,
        max_step=CFG.max_length,
        sos_token=1,
        eos_token=2,
        dropout=0.0,
        n_layers=2,
        num_heads=4,
    ):
        """
        num_classes:    number of possible characters to be outputted
        hidden_dim:     dimension of hidden states
        max_step:       maximum output sequence length
        sos_token:      start of sentence token
        eos_token:      end of sentence token
        dropout:        rate for regularization
        n_layers:       number of stacked LSTM layers
        num_heads:      number of parallel attention heads
        """
        super(Speller, self).__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.num_heads = num_heads

        self.max_step = max_step
        self.sos_token = sos_token
        self.eos_token = eos_token

        # Character embedding layer
        self.emb = nn.Embedding(self.num_classes, self.hidden_dim)

        # LSTM decoder
        self.rnn = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.n_layers,
            bidirectional=False,
            dropout=dropout,
            batch_first=True,
        )

        # Multi-head attention layer
        self.attention = MultiHeadAttention(self.hidden_dim, self.num_heads)

        # Linear layer for output prediction
        self.character_distribution = nn.Linear(self.hidden_dim, num_classes)

        # Activation function
        self.softmax = nn.LogSoftmax(dim=-1)

        self.init_rnn_weights()

    def init_rnn_weights(self, low=-0.1, high=0.1):
        for name, param in self.rnn.named_parameters():
            # Weights between input and hidden layers
            if "weight_ih" in name:
                torch.nn.init.uniform_(param.data, a=low, b=high)
            # Weights between hidden layers
            elif "weight_hh" in name:
                torch.nn.init.uniform_(param.data, a=low, b=high)
            elif "bias" in name:
                param.data.fill_(0)

    def init_hidden(self, batch_size):
        hidden = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_dim))
        cell = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_dim))
        return (hidden.to(CFG.DEVICE), cell.to(CFG.DEVICE))

    def forward_step(self, inputs, hc, listener_features):
        """
        process one timestep of decoding
        """
        decoder_output, hc = self.rnn(inputs, hc)
        att_out, context = self.attention(decoder_output, listener_features)
        logit = self.softmax(self.character_distribution(att_out))
        return logit, hc, context

    def forward(
        self,
        listener_features,
        ground_truth=None,
        teacher_forcing_rate=0.9,
        use_beam=False,
        beam_size=3,
    ):
        """
        The training uses Teacher Forcing method: it involves feeding observed
        sequence values (i.e. ground-truth samples) back into the RNN after each step
        thus forcing the RNN to stay close to the ground-truth sequence
        """
        if ground_truth is None:
            teacher_forcing_rate = 0
        teacher_forcing = True if np.random.random_sample() < teacher_forcing_rate else False

        if (ground_truth is None) and (not teacher_forcing):
            max_step = self.max_step
        else:
            max_step = ground_truth.size(1)

        input_word = torch.zeros(listener_features.size(0), 1).long().to(CFG.DEVICE)
        input_word[:, 0] = self.sos_token

        init_context = torch.zeros_like(listener_features[:, 0:1, :])
        inputs = self.emb(input_word)
        hc = self.init_hidden(input_word.size(0))
        logits = []

        if not use_beam:
            for step in range(max_step):
                logit, hc, context = self.forward_step(inputs, hc, listener_features)
                logits.append(logit.squeeze())
                if teacher_forcing:
                    output_word = ground_truth[:, step : step + 1]
                else:
                    output_word = logit.topk(1)[1].squeeze(-1)
                inputs = self.emb(output_word)

            logits = torch.stack(logits, dim=1)
            # y_hats = torch.max(logits, dim=-1)[1]
            return logits
        else:
            btz = listener_features.size(0)
            y_hats = torch.zeros(btz, max_step).long().to(CFG.DEVICE)
            logit, hc, context = self.forward_step(inputs, hc, listener_features)
            output_words = logit.topk(beam_size)[1].squeeze(1)
            for bi in range(btz):
                b_output_words = output_words[bi, :].unsqueeze(0).transpose(1, 0).contiguous()
                b_inputs = self.emb(b_output_words)
                b_listener_features = (
                    listener_features[bi, :, :]
                    .unsqueeze(0)
                    .expand((beam_size, -1, -1))
                    .contiguous()
                )
                if isinstance(hc, tuple):
                    b_h = hc[0][:, bi, :].unsqueeze(1).expand((-1, beam_size, -1)).contiguous()
                    b_c = hc[1][:, bi, :].unsqueeze(1).expand((-1, beam_size, -1)).contiguous()
                    b_hc = (b_h, b_c)
                else:
                    b_hc = hc[:, bi, :].unsqueeze(1).expand((-1, beam_size, -1)).contiguous()

                scores = torch.zeros(beam_size, 1).to(CFG.DEVICE)
                ids = torch.zeros(beam_size, max_step, 1).long().to(CFG.DEVICE)
                for step in range(max_step):
                    logit, b_hc, context = self.forward_step(b_inputs, b_hc, b_listener_features)
                    score, id = logit.topk(1)
                    scores += score.squeeze(1)
                    ids[:, step, :] = id.squeeze(1)
                    output_word = logit.topk(1)[1].squeeze(-1)
                    b_inputs = self.emb(output_word)
                y_hats[bi, :] = ids[scores.squeeze(1).topk(1)[1], :].squeeze(2)
            return y_hats
