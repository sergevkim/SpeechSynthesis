import einops
import torch
from torch import Tensor
from torch.nn import (
    Conv1d,
    ConvTranspose1d,
    Linear,
    Module,
    ModuleList,
    ReLU,
    Sequential,
    Softmax,
)


class CausalConv1d(Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            dilation: int,
        ):
        super().__init__()
        self.padding = dilation * (kernel_size - 1)
        self.conv = Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.padding,
        )

    def forward(
            self,
            inputs: Tensor,
        ) -> Tensor:
        outputs = self.conv(inputs)[:, :, :-self.padding]

        return outputs


class WaveNetBlock(Module):
    def __init__(
            self,
            dilation: int,
            mels_num: int = 80,
            skip_channels: int = 240,
            residual_channels: int = 120,
        ):
        super().__init__()
        self.conditional_conv = Conv1d(
            in_channels=mels_num,
            out_channels=residual_channels * 2,
            kernel_size=1,
        )
        self.causal_conv = CausalConv1d(
            in_channels=residual_channels,
            out_channels=residual_channels * 2,
            kernel_size=2,
            dilation=dilation,
        )
        self.skip_conv = Conv1d(
            in_channels=residual_channels,
            out_channels=skip_channels,
            kernel_size=1,
        )
        self.residual_conv = Conv1d(
            in_channels=residual_channels,
            out_channels=residual_channels,
            kernel_size=1,
        )

    def forward(
            self,
            inputs: Tensor,
            mel_specs: Tensor,
        ) -> Tensor:
        fir = self.conditional_conv(mel_specs)
        sec = self.causal_conv(inputs)

        hiddens = fir + sec
        residual_channels = hiddens.shape[1] // 2
        hiddens_1, hiddens_2 = torch.split(
            tensor=hiddens,
            split_size_or_sections=[residual_channels, residual_channels],
            dim=1,
        )
        hiddens = torch.tanh(hiddens_1) + torch.sigmoid(hiddens_2)

        skips = self.skip_conv(hiddens)
        residuals = self.residual_conv(hiddens) + inputs

        return skips, residuals


class WaveNetBody(Module):
    def __init__(
            self,
            blocks_num: int = 16,
            dilation_cycle_parameter: int = 8,
            audio_channels: int = 256,
            upsample_kernel_size: int = 800,
            win_length: int = 1024,
            hop_length: int = 256,
            mels_num: int = 80,
            skip_channels: int = 240,
            residual_channels: int = 120,
        ):
        super().__init__()
        upsample_stride = hop_length
        upsample_padding = (upsample_kernel_size
            + 4 * upsample_stride - win_length) // 2
        self.upsample_conv = ConvTranspose1d(
            in_channels=mels_num,
            out_channels=mels_num,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            padding=upsample_padding,
        )
        self.embedding_conv = Conv1d(
            in_channels=1,
            out_channels=residual_channels,
            kernel_size=1,
        )
        self.wavenet_blocks = ModuleList()

        for i in range(blocks_num // dilation_cycle_parameter):
            for j in range(dilation_cycle_parameter):
                self.wavenet_blocks.append(WaveNetBlock(
                    dilation=2 ** j,
                    mels_num=mels_num,
                    skip_channels=skip_channels,
                    residual_channels=residual_channels,
                ))

        self.head = Sequential(
            ReLU(),
            Conv1d(
                in_channels=skip_channels,
                out_channels=audio_channels,
                kernel_size=1,
            ),
            ReLU(),
            Conv1d(
                in_channels=audio_channels,
                out_channels=audio_channels,
                kernel_size=1,
            ),
            Softmax(),
        )

    def forward(
            self,
            waveforms: Tensor,
            mel_specs: Tensor,
        ) -> Tensor:
        mel_specs = self.upsample_conv(mel_specs)
        residuals = self.embedding_conv(waveforms)

        skips_list = list()

        for block in self.wavenet_blocks:
            skips, residuals = block(
                inputs=residuals,
                mel_specs=mel_specs,
            )
            skips_list.append(skips)

        skips = torch.stack(skips_list, dim=0).sum(dim=0)
        skips = einops.reduce(
            tensor=skips,
            pattern='batch_size skip_channels length -> skip_channels length',
            reduction='sum',
        )
        outputs = self.head(skips)

        return outputs


if __name__ == '__main__':
    wavenet = WaveNetBody()
    print(wavenet)
