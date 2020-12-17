from pathlib import Path
from typing import List, Union


class CharVocab:
    def __init__(self):
        self.char2tag = dict()
        self.tag2char = dict()
        self.char2count = dict()
        self.chars_num = 0

        self.add_char('<pad>')
        self.add_char('<sos>')
        self.add_char('<eos>')

    def __len__(self) -> int:
        return len(self.char2tag)

    def __str__(self) -> str:
        return str(self.char2tag)

    def add_char(
            self,
            char: str,
        ) -> None:
        if char not in self.char2tag:
            tag = self.chars_num
            self.char2tag[char] = tag
            self.char2count[char] = 1
            self.tag2char[tag] = char
            self.chars_num += 1
        else:
            self.char2count[char] += 1

    @staticmethod
    def get_lines(
            text_corpus_path: Union[Path, str],
        ) -> List[str]:
        with open(text_corpus_path) as f:
            lines = f.readlines()

        return lines

    @staticmethod
    def lines2char_seqs(
            lines: List[str],
        ) -> List[List[str]]:
        char_seqs = list()

        for line in lines:
            char_seq = list()
            char_seq.append('<sos>')
            char_seq += list(line)
            char_seq.append('<eos>')

            char_seqs.append(char_seq)

        return char_seqs

    def char_seqs2tag_seqs(
            self,
            char_seqs: List[List[str]]
        ) -> List[List[int]]:
        tag_seqs = list()

        for char_seq in char_seqs:
            tag_seq = list()

            for char in char_seq:
                tag = self.char2tag[char]
                tag_seq.append(tag)

            tag_seqs.append(tag_seq)

        return tag_seqs

    @staticmethod
    def prepare_lj_speech_lines(
            lines: List[str],
        ) -> List[str]:
        prepared_lines = list()

        for line in lines:
            prepared_line = line.split('|')[-1][:-1].lower()
            prepared_lines.append(prepared_line)

        return prepared_lines

    @classmethod
    def build_char_vocab(
            cls,
            text_corpus_path: Union[Path, str],
        ):
        char_vocab = CharVocab()

        lines = cls.get_lines(text_corpus_path=text_corpus_path)
        lines = cls.prepare_lj_speech_lines(lines=lines)
        char_seqs = cls.lines2char_seqs(lines=lines)

        for char_seq in char_seqs:
            for char in char_seq:
                char_vocab.add_char(char=char)

        return char_vocab

