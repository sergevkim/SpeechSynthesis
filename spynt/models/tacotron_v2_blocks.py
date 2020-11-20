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


class Tacotron2Encoder(Module):
    def __init__(
            self,
            dict_length: int,
            embedding_dim: int=512,
        ):
        super().__init__()
        self.embedding = Embedding(
            num_embeddings=dict_length,
            embedding_dim=embedding_dim,
        ),
        self.convs_ordered_dict = OrderedDict()

        for i in range(3):
            self.convs_ordered_dict[f'conv_block_{i}'] = Sequential(
                Conv1d(
                    in_channels=embedding_dim,
                    out_channels=embedding_dim,
                    kernel_size=5,
                ),
                BatchNorm1d(embedding_dim),
                ReLU(),
                Dropout(p=0.5),
            )

        self.convs = Sequential(self.convs_ordered_dict)
        self.lstm = LSTM(
            input_size=embedding_num,
            hidden_size=(embedding_num / 2),
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

    def forward(
            self,
            x: Tensor,
            input_lengths: int,
        ):
        embedded_x = self.embedding(x)
        embedded_x = einops.rearrange(embedded_x, 'b l -> l b')

        convs_outputs = self.convs(embedded_x)
        convs_outputs = einops.rearrange(convs_outputs, 'l b -> b l')

        lstm_outputs, _ = self.lstm(convs_outputs)

        return lstm_outputs


class DecoderPreNet(Module):
    def __init__(
            self,
            in_features: int=256,
            out_features: int=256,
            dropout_p: float=0.5,
        ):
        super().__init__()

        blocks_ordered_dict = OrderedDict()

        for i in range(2):
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
        ):
        super().__init__()

        blocks_ordered_dict = OrderedDict()

        for i in range(5):
            blocks_ordered_dict[f'conv_block_{i}'] = Sequential(
                Conv1d(), #TODO
                BatchNorm1d(),
            )

        self.postnet_sequential = Sequential(blocks_ordered_dict)

    def forward(
            self,
            x: Tensor,
        ) -> Tensor:
        return self.postnet_sequential(x)


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

