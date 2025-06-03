from torch import nn


class LAS(nn.Module):
    """
    Listen, Attend, and Spell model
    """

    def __init__(self, listener, speller):
        super(LAS, self).__init__()
        self.encoder = listener
        self.decoder = speller

    def forward(
        self,
        inputs,
        ground_truth=None,
        teacher_forcing_rate=0.9,
        use_beam=False,
        beam_size=16,
    ):
        listener_features, hidden = self.encoder(inputs)
        logits = self.decoder(
            listener_features,
            ground_truth,
            teacher_forcing_rate=teacher_forcing_rate,
            use_beam=use_beam,
            beam_size=beam_size,
        )

        return logits
