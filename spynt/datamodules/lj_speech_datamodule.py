from pathlib import Path
from PIL import Image
import string
from typing import List, Tuple

import einops
import torch
import torchaudio
from torch import Tensor, FloatTensor, IntTensor
from torch.utils.data import Dataset, DataLoader

from spynt.utils import CharVocab


def zero_padding(sequence, new_length):
    padded_sequence = torch.zeros(new_length)
    padded_sequence[:new_length] = sequence[:new_length]

    return padded_sequence


class LJSpeechDataset(Dataset):
    def __init__(
            self,
            filenames: List[str],
            tokens_seqs: List[List[str]],
            max_waveform_length: int=20000,
            max_target_length: int=100,
        ):
        self.filenames = filenames
        self.tokens_seqs = tokens_seqs

    def get_tags(
            self,
            idx: int,
        ) -> IntTensor:
        tokens_seq = self.tokens_seqs[idx]
        tags_seq = Tensor(Vocabulary.tokens_seq2tags_seq(
            tokens_seq=tokens_seq,
        ))
        #target_length = min(len(target), self.max_target_length)
        #padded_target = torch.zeros(self.max_target_length)
        #padded_target[:target_length] = target[:target_length]
        #target_length = torch.tensor(target_length)

        return tags_seq

    def get_waveform(
            self,
            idx: int,
        ) -> Tuple[Tensor, Tensor]:
        filename = self.filenames[idx]
        waveform, sample_rate = torchaudio.load(filename)
        waveform = einops.rearrange(waveform, 'b x -> (b x)')

        waveform_length = min(len(waveform), self.max_waveform_length)
        padded_waveform = torch.zeros(self.max_waveform_length)
        padded_waveform[:waveform_length] = waveform[:waveform_length]
        waveform_length = torch.tensor(waveform_length)

        return padded_waveform, waveform_length

    def __getitem__(
            self,
            idx: int,
        ):

        waveform, waveform_length = self.get_waveform(idx=idx)
        tags_seq, tags_seq_length = self.get_tags(idx=idx)

        result = (
            waveform,
            tags_seq,
            waveform_length,
            tags_seq_length,
        )

        return result

    def __len__(self):
        return len(self.filenames)


class LJSpeechDataModule:
    def __init__(
            self,
            data_path: Path,
            batch_size: int,
            num_workers: int,
        ):
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        wavs_path = self.data_path / "wavs"
        targets_path = self.data_path / "metadata.csv"
        wav_filenames = list(str(p) for p in wavs_path.glob('*.wav'))
        wav_filenames.sort()
        targets_file = open(targets_path, 'r')

        targets = list()

        for i in range(len(wav_filenames)):
            line = targets_file.readline()
            table = str.maketrans('', '', string.punctuation)
            target = line.split('|')[-1].lower().translate(table)[:-1]
            targets.append(target)

        data = dict(
            filenames=wav_filenames,
            targets=targets,
        )

        return data

    def setup(
            self,
            val_ratio: float,
        ) -> None:
        data = self.prepare_data()
        wav_filenames = data['filenames']
        targets = data['targets']

        full_dataset = LJSpeechDataset(
            filenames=wav_filenames,
            targets=targets,
        )

        full_size = len(full_dataset)
        val_size = int(val_ratio * full_size)
        train_size = full_size - val_size

        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            dataset=full_dataset,
            lengths=[train_size, val_size],
        )

    def train_dataloader(self) -> DataLoader:
        train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

        return train_dataloader

    def val_dataloader(self) -> DataLoader:
        val_dataloader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

        return val_dataloader

    def test_dataloader(self):
        pass

