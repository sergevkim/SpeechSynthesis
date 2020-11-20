import torch
from torch import Tensor

from .tacotron_v2_blocks import (
    Tacotron2Decoder,
    Tacotron2Encoder,
    WaveNetVocoder,
)


class Tacotron2Synthesizer(Module):
    def __init__(
            self,
            dict_length: int,
            embedding_dim: int=512,
        ):
        super().__init__()
        self.encoder = Tacotron2Encoder(
            dict_length=dict_length,
            embedding_dim=embedding_dim,
        )
        self.decoder = Tacotron2Decoder()
        self.vocoder = WaveNetVocoder()

    def forward(
            self,
            x: Tensor,
        ) -> Tensor:
        encoder_outputs = self.encoder(x)
        decoder_outputs = self.decoder(encoder_outputs)
        vocoder_outputs = self.vocoder(decoder_outputs)

        return vocoder_outputs

    def training_step(self):
        pass

    def validation_step(self):
        pass

