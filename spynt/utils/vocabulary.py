from __future__ import annotations
from pathlib import Path
import string
from typing import List, Union


class Vocabulary:
    def __init__(self):
        self.token2tag = dict()
        self.tag2token = dict()
        self.token2count = dict()
        self.tokens_num = 0

        self.add_token('<pad>')
        self.add_token('<sos>')
        self.add_token('<eos>')

    def __len__(self) -> int:
        return len(self.token2tag)

    def __str__(self) -> str:
        return str(self.token2tag)

    def add_token(
            self,
            token: str,
        ) -> None:
        if token not in self.token2tag:
            tag = self.tokens_num
            self.token2tag[token] = tag
            self.token2count[token] = 1
            self.tag2token[tag] = token
            self.tokens_num += 1
        else:
            self.token2count[token] += 1

    @staticmethod
    def get_lines(
            text_corpus_path: Union[Path, str],
        ) -> List[str]:
        with open(text_corpus_path) as f:
            lines = f.readlines()

        return lines

    @staticmethod
    def lines2token_seqs(
            lines: List[str],
        ) -> List[List[str]]:
        token_seqs = list()

        for line in lines:
            token_seq = list()
            token_seq.append('<sos>')
            token_seq += line.split()
            token_seq.append('<eos>')

            token_seqs.append(token_seq)

        return token_seqs

    def token_seqs2tag_seqs(
            self,
            token_seqs: List[List[str]]
        ) -> List[List[int]]:
        tag_seqs = list()

        for token_seq in token_seqs:
            tag_seq = list()

            for token in token_seq:
                tag = self.token2tag[token]
                tag_seq.append(tag)

            tag_seqs.append(tag_seq)

        return tag_seqs

    @classmethod
    def prepare_lj_speech_lines(
            cls,
            lines: List[str],
        ) -> List[str]:
        prepared_lines = list()

        for line in lines:
            prepared_line = line.split('|')[-1].lower()

            for p in string.punctuation:
                prepared_line = prepared_line.replace(p, f' {p} ')

            prepared_lines.append(prepared_line)

        return prepared_lines

    @classmethod
    def build_vocabulary(
            cls,
            text_corpus_path: Union[Path, str],
        ) -> Vocabulary:
        vocabulary = Vocabulary()

        lines = cls.get_lines(text_corpus_path=text_corpus_path)
        lines = cls.prepare_lj_speech_lines(lines=lines)
        token_seqs = cls.lines2token_seqs(lines=lines)

        for token_seq in token_seqs:
            for token in token_seq:
                vocabulary.add_token(token=token)

        return vocabulary

