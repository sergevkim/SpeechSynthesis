from torch.nn import (
    Linear,
    Module,
    Sequential,
)


class WaveNetComponent(Module):
    def __init__(
            self,
        ):
        super().__init__()
        self.wavenet_component = Linear(3, 4)

    def forward(
            self,
            x: Tensor
        ) -> Tensor:
        x = self.wavenet_component(x)

        return x

