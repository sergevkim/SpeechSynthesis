from collections import OrderedDict

import einops
import torch
from torch import Tensor
from torch.nn import (
    BatchNorm1d,
    Conv1d,
    Dropout,
    Embedding,
    Linear,
    LSTM,
    Module,
    ReLU,
)
import torch.nn.utils.rnn as rnn


class Tacotron2Encoder(Module):
    def __init__(
            self,
            dict_length: int,
            convs_num: int=3,
            embedding_dim: int=512,
            kernel_size: int=5,
            dropout_p: float=0.5,
        ):
        super().__init__()
        self.embedding = Embedding(
            num_embeddings=dict_length,
            embedding_dim=embedding_dim,
        ),
        self.convs_ordered_dict = OrderedDict()

        for i in range(convs_num):
            self.convs_ordered_dict[f'conv_block_{i}'] = Sequential(
                Conv1d(
                    in_channels=embedding_dim,
                    out_channels=embedding_dim,
                    kernel_size=kernel_size,
                ),
                BatchNorm1d(embedding_dim),
                ReLU(),
                Dropout(p=dropout_p),
            )

        self.convs = Sequential(self.convs_ordered_dict)
        self.lstm = LSTM(
            input_size=embedding_num,
            hidden_size=(embedding_num // 2),
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

    def forward(
            self,
            x: Tensor,
            input_lengths: List[int], #TODO np.array?
        ):
        embedded_x = self.embedding(x)
        embedded_x = einops.rearrange(
            tensor=embedded_x,
            pattern='b l e -> b e l',
        )

        convs_outputs = self.convs(embedded_x)
        convs_outputs = einops.rearrange(
            tensor=convs_outputs,
            pattern='b e l -> b l e',
        )
        packed_convs_outputs = rnn.pack_padded_sequence(
            input=convs_outputs,
            length=input_lengths,
            batch_first=True,
            enforce_sorted=False,
        )

        lstm_outputs, _ = self.lstm(packed_convs_outputs)
        padded_lstm_outputs, _ = rnn.pad_packed_sequence(
            sequence=lstm_outputs,
            batch_first=True,
        )

        return padded_lstm_outputs


class DecoderPreNet(Module):
    def __init__(
            self,
            in_features: int=256,
            out_features: int=256,
            dropout_p: float=0.5,
            blocks_num: int=2,
        ):
        super().__init__()

        blocks_ordered_dict = OrderedDict()

        for i in range(blocks_num):
            blocks_ordered_dict[f'linear_block_{i}'] = Sequential(
                Linear(
                    in_features=in_features,
                    out_features=out_features,
                ),
                ReLU(),
                Dropout(p=dropout_p),
            )

        self.prenet_sequential = Sequential(blocks_ordered_dict)

    def forward(
            self,
            x: Tensor,
        ) -> Tensor:
        return self.prenet_sequential(x)


class DecoderPostNet(Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int=5,
            blocks_num: int=5,
        ):
        super().__init__()

        blocks_ordered_dict = OrderedDict()

        blocks_ordered_dict['conv_block_0'] = Sequential(
            Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            BatchNorm1d(num_features=out_channels),
        )

        for i in range(1, blocks_num):
            blocks_ordered_dict[f'conv_block_{i}'] = Sequential(
                Conv1d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                ),
                BatchNorm1d(num_features=out_channels),
            )

        self.postnet_sequential = Sequential(blocks_ordered_dict)

    def forward(
            self,
            x: Tensor,
        ) -> Tensor:
        x = einops.rearrange(
            tensor=x,
            pattern='b f m -> b m f',
        )
        postnet_outputs = self.postnet_sequential(x)
        postnet_outputs = einops.rearrange(
            tensor=postnet_outputs,
            pattern='b f m -> b m f',
        )

        return postnet_outputs


class Tacotron2Decoder(Module):
    def __init__(
            self,
        ):
        super().__init__()
        self.prenet = DecoderPreNet()
        self.postnet = DecoderPostNet()

    def forward(
            self,
            x: Tensor,
        ) -> Tensor:
        pass


class WaveNetVocoder(Module):
    def __init__(
            self,
        ):
        super().__init__()

    def forward(
            self,
            x: Tensor,
        ):
        pass

