from collections import OrderedDict
from typing import Tuple

import einops
import torch
from torch import Tensor
from torch.nn import (
    BatchNorm1d,
    Conv1d,
    Dropout,
    Embedding,
    Linear,
    LSTMCell,
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


class PreNet(Module):
    def __init__(
            self,
            in_features: int=80,
            out_features: int=256,
            dropout_p: float=0.5,
            blocks_num: int=2,
        ):
        super().__init__()

        blocks_ordered_dict = OrderedDict()
        blocks_ordered_dict['linear_block_0'] = Sequential(
            Linear(
                in_features=in_features,
                out_features=out_features,
            ),
            ReLU(),
            Dropout(p=dropout_p),
        )

        for i in range(1, blocks_num):
            blocks_ordered_dict[f'linear_block_{i}'] = Sequential(
                Linear(
                    in_features=out_features,
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


class LocationBlock(nn.Module):
    def __init__(
            self,
            attention_n_filters: int,
            attention_kernel_size: int,
            attention_dim: int,
        ):
        super().__init__()

        padding = attention_kernel_size // 2

        blocks_ordered_dict = OrderedDict()
        blocks_ordered_dict['conv'] = Conv1d(
            in_channels=2,
            out_channels=attention_n_filters,
            kernel_size=attention_kernel_size,
            padding=padding,
            bias=False,
            stride=1,
            dilation=1,
        )
        blocks_ordered_dict['transpose'] = Rearrange(
            pattern='b x y -> b y x',
        )
        blocks_ordered_dict['projection'] = Linear(
            in_features=attention_n_filters,
            out_features=attention_dim,
            bias=False,
        )

        self.blocks = Sequential(blocks_ordered_dict)

    def forward(self, attention_weights):
        output = self.blocks(attention_weights)

        return output


class LocationSensitiveAttention(nn.Module):
    def __init__(
            self,
            attention_rnn_dim,
            embedding_dim,
            attention_dim,
            attention_location_n_filters,
            attention_location_kernel_size
        ):
        super().__init__()

        self.query = nn.Linear(attention_rnn_dim, attention_dim, bias=False)
        self.memory = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.b = nn.Linear(attention_dim, 1, bias=False)
        self.location_layer = LocationBlock(
            attention_location_n_filters,
            attention_location_kernel_size,
            attention_dim
        )
        self.score_mask_value = -float("inf")

    def get_alignment_energies(
            self,
            query,
            processed_memory,
            attention_weights_cat
        ):
        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_weights_cat)

        energies = self.v(torch.tanh(
            processed_query + processed_attention_weights + processed_memory
        ))

        energies = energies.squeeze(2)

        return energies

    def forward(
            self,
            attention_hidden_state,
            memory,
            processed_memory,
            attention_weights_cat,
            mask
        ):
        alignment = self.get_alignment_energies(
            attention_hidden_state, processed_memory, attention_weights_cat
        )

        alignment = alignment.masked_fill(mask, self.score_mask_value)

        attention_weights = F.softmax(alignment, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights


class Tacotron2Decoder(Module):
    def __init__(
            self,
            mel_channels_num: int=80,
            frames_per_step_num: int=1,
            embedding_dim: int=512,
            prenet_dim: int=256,
            prenet_dropout_p: float=0.5,
            prenet_blocks_num: int=2,
            attention_rnn_dim: int=512,
            decoder_rnn_dim: int=512,
        ):
        super().__init__()

        self.prenet = PreNet(
            in_features=mel_channels_num * frames_per_step_num,
            out_features=prenet_dim,
            dropout_p=prenet_dropout_p,
            blocks_num=prenet_blocks_num,
        )
        self.attention_rnn = LSTMCell(
            input_size=embedding_dim + prenet_dim,
            hidden_size=attention_rnn_dim,
        )
        self.attention_layer = Attention(
            #TODO
        )
        self.decoder_rnn = LSTMCell(
            input_size=embedding_dim + attention_rnn_dim,
            hidden_size=decoder_rnn_dim,
        )
        self.linear_projection = Linear(
            in_features=decoder_rnn_dim + encoder_embedding_dim,
            out_features=mel_channels_num * frames_per_step_num,
        )
        self.stop_token_linear_projection = Linear(
            in_features=embedding_dim + attention_rnn_dim,
            out_features=1,
        )

    def init_zero_state(
            self,
            batch_size: int,
            device: torch.device,
        ) -> Dict[str, Tensor]:
        decoder_outputs = torch.zeros(
            size=(
                batch_size,
                self.mel_channels_num * self.frames_per_step_num,
            ),
        ).to(self.device)

        state = dict(
            decoder_outputs=decoder_outputs,
        )

        return state

    def forward(
            self,
            encoder_outputs: Tensor,
            lengths: Tensor,
            mel_specs: Tensor, #TODO check types with torch
        ) -> Tensor:
        batch_size, _, _ = encoder_outputs.shape
        zero_state = self.init_zero_state(batch_size=batch_size)

        decoder_outputs = zero_state['decoder_outputs']


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


class PostNet(Module):
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
            pattern='b m f -> b f m',
        )

        return postnet_outputs

