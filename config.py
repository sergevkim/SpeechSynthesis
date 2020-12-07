from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class CommonArguments:
    device: torch.device = torch.device('cuda:0')
    seed: int = 9
    verbose: bool = True
    version: str = '0.1.0'


@dataclass
class DataArguments:
    batch_size: int = 2
    data_path: Path = Path('./data/LJSpeech-1.1')
    learning_rate: float = 3e-4
    num_workers: int = 4
    val_ratio: float = 0.1


@dataclass
class TrainArguments:
    max_epoch: int = 10
    one_batch_overfit: bool = True
    scheduler_gamma: float = 0.5
    scheduler_step_size: int = 10


@dataclass
class SpecificArguments:
    n_fft: int
    win_length: int
    hop_length: int
    vocoder_audio_channels: int = 256
    vocoder_dilation_cycle_parameter: int = 8
    vocoder_repeated_layers: int = 16
    vocoder_residual_channels: int = 120
    vocoder_skip_channels: int = 240
