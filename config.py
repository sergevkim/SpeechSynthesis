from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class Arguments:
    batch_size: int = 64
    data_path: Path = Path('/Users/sergevkim/git/sergevkim/SpeechRecognition/data/LJSpeech-1.1')
    device: torch.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    learning_rate: float = 3e-4
    max_epoch: int = 10
    num_workers: int = 4
    val_ratio: float = 0.1
    verbose: bool = True
    version: str = '0.1.0'

