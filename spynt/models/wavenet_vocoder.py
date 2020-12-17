from typing import List, Tuple

from torch import Tensor
from torch.nn import Module
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import StepLR, _LRScheduler
from torchaudio.trandforms import (
    MuLawEncoding,
    MuLawDecoding,
)

from spynt.models.wavenet_components import WaveNetBody


class WaveNetVocoder(Module):
    def __init__(
            self,
            device,
            learning_rate: float = 3e-4,
            verbose: bool = True,
            version: str = '1.0',
        ):
        super().__init__()
        self.device = device
        self.learning_rate = learning_rate
        self.scheduler_step_size = 1
        self.scheduler_gamma = 1
        self.verbose = verbose
        self.version = version

        self.mel_spectrogramer = MelSpectrogram(
            n_mels=80,
        ).to(device)
        self.mu_law_encoder = MuLawEncoding(256)
        self.mu_law_decoder = MuLawDecoding(256)

        self.wavenet_body = WaveNetBody()

    def forward(
            self,
            x: Tensor
        ) -> Tensor:
        x = self.wavenet_body(x)

        return x

    def training_step(
            self,
            batch: Tensor,
            batch_idx: int,
        ) -> Tensor:
        waveforms, _ = batch
        waveforms = waveforms.to(device)
        mel_specs = self.mel_spectrogramer(waveforms)

        outputs, ground_truths = self.wavenet_body(
            waveforms=waveforms,
            mel_specs=mel_specs,
        )
        loss = self.criterion(outputs, ground_truths)

        return loss

    def training_step_end(self):
        pass

    def training_epoch_end(
            self,
            epoch_idx: int,
        ) -> None:
        if self.verbose:
            print(f"Training epoch {epoch_idx} is over.")

    def validation_step(
            self,
            batch: Tensor,
            batch_idx: int,
        ) -> Tensor:
        loss = self.training_step(batch, batch_idx)

        return loss

    def validation_step_end(self):
        pass

    def validation_epoch_end(
            self,
            epoch_idx: int,
        ) -> None:
        if self.verbose:
            print(f"Validation epoch {epoch_idx} is over.")

    def configure_optimizers(
            self,
        ) -> Tuple[List[Optimizer], List[_LRScheduler]]:
        optimizer = Adam(
            params=self.wavenet_body.parameters(),
            lr=self.learning_rate,
        )
        optimizers = [optimizer]

        scheduler = StepLR(
            optimizer=optimizer,
            step_size=self.scheduler_step_size,
            gamma=self.scheduler_gamma,
        )
        schedulers = []

        return optimizers, schedulers


if __name__ == '__main__':
    vocoder = WaveNetVocoder()
    print(vocoder)
