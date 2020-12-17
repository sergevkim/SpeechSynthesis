from typing import List, Tuple

from torch import Tensor
from torch.nn import Module
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import StepLR, _LRScheduler

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
        self.verbose = verbose
        self.version = version

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
        loss = self(batch)

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
        loss = self(batch)

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
