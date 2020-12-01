import torch
from torch import Tensor
from torch.nn import MSELoss

from .tacotron_v2_blocks import (
    PostNet,
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
        self.criterion = MSELoss()

        self.encoder = Tacotron2Encoder(
            dict_length=dict_length,
            embedding_dim=embedding_dim,
        )
        self.decoder = Tacotron2Decoder()
        self.postnet = PostNet()
        self.vocoder = WaveNetVocoder()

    def forward(
            self,
            tags: Tensor,
            lengths: Tensor,
            mel_specs: Tensor,
        ) -> Tensor:
        encoder_outputs = self.encoder(tags, lengths)
        decoder_mel_specs, decoder_probs = self.decoder(
            encoder_outputs,
            lengths,
            mel_specs,
        )
        postnet_melspecs = self.postnet(decoder_mel_specs)
        outputs = decoder_mel_specs + postnet_mel_specs

        return outputs

    def training_step(
            self,
            batch: Tensor,
            batch_idx: int,
        ) #TODO:
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        y_hat = self(x)
        loss = criterion(y_hat, y)

        return loss

    def validation_step(self):
        pass

