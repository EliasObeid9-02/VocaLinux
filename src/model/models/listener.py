import torch
from torch import nn
from torch.autograd import Variable

from model.config import CFG


class BLSTM(nn.Module):
    """
    Bidrectional Long Short Term Memory RNN module
    The module uses LSTM in order to rememeber the previous sounds to get
    a more accurate prediction of what the current letter should be
    """

    def __init__(
        self, input_dim, hidden_dim, dropout=0.0, n_layers=1, bidirectional=True
    ):
        """
        input_dim:      dimensions of input features
        hidden_dim:     dimensions of LSTM hidden state
        dropout:        rate for regularization
        n_layers:       number of stacked LSTM layers
        bidirectional:  enables bidirectional processing if true
        """
        super(BLSTM, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional

        self.rnn = nn.LSTM(
            input_size=self.input_dim * 2,
            hidden_size=self.hidden_dim,
            num_layers=self.n_layers,
            bidirectional=self.bidirectional,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, inputs):
        if isinstance(inputs, tuple):
            inputs, hc = inputs

        batch_size = inputs.size(0)
        seq_len = inputs.size(1)
        input_size = inputs.size(2)

        # Padding with zeroes to guarantee even length
        if seq_len % 2:
            zeros = torch.zeros((inputs.size(0), 1, inputs.size(2))).to("cpu")
            inputs = torch.cat([inputs, zeros], dim=1)
            seq_len += 1

        # Downsampling the input sequence size
        inputs = inputs.contiguous().view(batch_size, int(seq_len / 2), input_size * 2)

        output, hc = self.rnn(inputs, hc)
        return (output, hc)


class Listener(nn.Module):
    """
    The Listener module is a pyramid made up of three layers of BLSTM networks
    each layer in the pyramid reduces the length of the input by a factor of 2
    allowing the following Attention module to extract information more efficiently
    """

    def __init__(
        self, input_dim, hidden_dim, dropout=0.0, n_layers=1, bidirectional=True
    ):
        """
        input_dim:      dimensions of input features
        hidden_dim:     dimensions of LSTM hidden state
        dropout:        rate for regularization
        n_layers:       number of stacked LSTM layers per BLSTM block
        bidirectional:  enables bidirectional processing if true
        """
        super(Listener, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional

        # Constructing the 3 layer pyramid of BLSTM
        self.pblstm = nn.Sequential(
            BLSTM(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                dropout=dropout,
                n_layers=n_layers,
                bidirectional=self.bidirectional,
            ),
            BLSTM(
                input_dim=self.hidden_dim * 2 if self.bidirectional else 0,
                hidden_dim=self.hidden_dim,
                dropout=dropout,
                n_layers=n_layers,
                bidirectional=self.bidirectional,
            ),
            BLSTM(
                input_dim=self.hidden_dim * 2 if self.bidirectional else 0,
                hidden_dim=self.hidden_dim,
                dropout=dropout,
                n_layers=n_layers,
                bidirectional=self.bidirectional,
            ),
        )

    def init_hidden(self, batch_size):
        """
        Initialiazing the hidden and cell states for each LSTM
        using the batch_size
        """
        hidden = Variable(
            torch.zeros(
                self.n_layers * 2 if self.bidirectional else 1,
                batch_size,
                self.hidden_dim,
            )
        )
        cell = Variable(
            torch.zeros(
                self.n_layers * 2 if self.bidirectional else 1,
                batch_size,
                self.hidden_dim,
            )
        )
        return (hidden.to(CFG.DEVICE), cell.to(CFG.DEVICE))

    def forward(self, inputs):
        hc = self.init_hidden(inputs.size(0))
        output, state = self.pblstm((inputs, hc))
        return output, state
