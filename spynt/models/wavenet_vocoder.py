from torch.nn import Module

from spynt.models.wavenet_components import WaveNetComponent


class WaveNetVocoder(Module):
    def __init__(
            self,
        ):
        super().__init__()
        self.wavenet_component = WavenetComponent()

    def forward(
            self,
            x: Tensor
        ) -> Tensor:
        x = self.wavenet_component(x)

        return x


if __name__ == '__main__':
    vocoder = WaveNetVocoder()
    print(vocoder)

