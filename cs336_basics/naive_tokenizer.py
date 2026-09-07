import json
import os
import this
from typing import Iterable, Iterator, Any

import msgspec
import regex

from .ran_train_bpe import *
import regex as re

import base64


class NaiveTokenizer:

    def __init__(self, vocab: dict[int, bytes],
                 merges: list[tuple[bytes, bytes]],
                 special_tokens: list[str] | None = None, ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

    def encode(self, text: str) -> list[int]:
        vocab_quick_reference: dict[bytes, int] = {
            token_bytes: token_id
            for token_id, token_bytes in self.vocab.items()
        }

        result: list[int] = []

        # Split while preserving special tokens.
        if self.special_tokens:
            special_pattern = "|".join(
                re.escape(token)
                for token in sorted(
                    self.special_tokens,
                    key=len,
                    reverse=True,
                )
            )

            # split_chunks = [re.findall(PAT, split_chunk) for split_chunk in split_chunks if len(split_chunks) > 0]
            special_tokens_set = set(self.special_tokens)
            split_chunks = list()
            for chunk in re.split(f"({special_pattern})", text):
                if chunk in special_tokens_set:
                    split_chunks.append(chunk)
                    continue
                split_chunks.extend(re.findall(PAT, chunk))
        else:
            split_chunks = re.findall(PAT, text)
            special_tokens_set = set()
        for chunk in split_chunks:
            if not chunk:
                continue

            if chunk in special_tokens_set:
                token_bytes = chunk.encode("utf-8")
                result.append(
                    vocab_quick_reference[token_bytes]
                )
                continue
            result.extend(self.encode_chunks(chunk, vocab_quick_reference))
        return result

    def encode_chunks(self, chunk: str, vocab_quick_reference: dict[bytes, int]) -> list[int]:
        chunk_ids = [
            vocab_quick_reference[bytes([b])]
            for b in chunk.encode("utf-8")
        ]

        for left_bytes, right_bytes in self.merges:
            left_id = vocab_quick_reference[left_bytes]
            right_id = vocab_quick_reference[right_bytes]

            merged_bytes = left_bytes + right_bytes
            merged_id = vocab_quick_reference[merged_bytes]
            chunk_result: list[int] = []
            idx = 0

            while idx < len(chunk_ids):
                if idx + 1 < len(chunk_ids) and chunk_ids[idx] == left_id and chunk_ids[idx + 1] == right_id:
                    chunk_result.append(merged_id)
                    idx += 2
                else:
                    chunk_result.append(chunk_ids[idx])
                    idx += 1

            chunk_ids = chunk_result
        return chunk_ids

    def decode(self, ids: list[int]) -> str:
        result = bytearray()

        for token_id in ids:
            if token_id not in self.vocab:
                raise ValueError(
                    f"token_id {token_id} not in vocab"
                )

            result.extend(self.vocab[token_id])

        return result.decode("utf-8", errors="replace")

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        special_pattern = None
        special_tokens_set = set()
        vocab_quick_reference: dict[bytes, int] = {
            token_bytes: token_id
            for token_id, token_bytes in self.vocab.items()
        }
        if self.special_tokens:
            special_pattern = "|".join(
                re.escape(token)
                for token in sorted(
                    self.special_tokens,
                    key=len,
                    reverse=True,
                )
            )
            special_tokens_set = set(self.special_tokens)
        for line in iterable:
            text = line
            for line in iterable:
                text += line
                if special_pattern is not None:
                    if re.match(special_pattern, line):
                        break

            splits = [text]
            if special_pattern is not None:
                splits = re.split(f"({special_pattern})", text)
            for idx, chunk in enumerate(splits):
                if chunk in special_tokens_set:
                    yield vocab_quick_reference[chunk.encode("utf-8")]
                    continue
                for chunk_part in re.findall(PAT, chunk):
                    for token_id in self.encode_chunks(chunk_part, vocab_quick_reference):
                        yield token_id
class JSONEncodeData(msgspec.Struct):
    vocab: dict[int, bytes]
    merges: list[tuple[bytes, bytes]]
    special_tokens: list[str]

    @classmethod
    def from_tokenizer(cls, tokenizer: "NaiveTokenizer") -> "JSONEncodeData":
        return cls(
            vocab=tokenizer.vocab,
            merges=tokenizer.merges,
            special_tokens=tokenizer.special_tokens,
        )