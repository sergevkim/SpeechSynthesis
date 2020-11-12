from collections import OrderedDict

import torch
from torch import Tensor
from torch.nn import (
    BatchNorm1d,
    Conv1d,
    Embedding,
    Linear,
    LSTM,
    Module,
)


class WaveNetVocoder(Module):
    def __init__(
            self,
        ):
        pass

    def forward(
            self,
            x: Tensor,
        ):
        pass


class Tacotron2Synthesizer(Module):
    def __init__(
            self,
            dict_length: int,
            embedding_dim: int=512,
        ):
        self.encoder = Sequential(OrderedDict(
            embedding=Embedding(
                num_embeddings=dict_length,
                embedding_dim=embedding_dim,
            ),
            convs=Sequential(
                [
                    Conv1d(
                        in_channels=embedding_dim,
                        out_channels=embedding_dim,
                        kernel_size=5,
                    ),
                    BatchNorm1d(embedding_dim)
                    for i in range(3)
                ]
            ),
            lstm=LSTM(
                input_size=embedding_num,
                hidden_size=(embedding_num / 2),
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
        ))
        self.decoder = Sequential()
        self.vocoder = WaveNetVocoder()

    def forward(
            self,
            x_0: Tensor,
        ):
        x_1 = self.encoder(x_0)
        x_2 = self.decoder(x_1)
        x_3 = self.vocoder(x_2)

        return x_3

