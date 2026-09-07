import os

import msgspec
import numpy as np
import torch

import cs336_basics.naive_tokenizer
import cs336_basics.ran_train_bpe
from cs336_basics import naive_tokenizer, ran_train_bpe

TOKENIZER_STORE_FILE = "TOKENIZER_STORE_FILE"
TOKENIZED_FILE_SUFFIX = "bin"


def new_from_state_file(state_file: str | os.PathLike) -> cs336_basics.naive_tokenizer.NaiveTokenizer:
    with open(state_file, "rb") as f:
        json_state = f.read()
        result = msgspec.json.decode(json_state, type=cs336_basics.naive_tokenizer.JSONEncodeData)
        return cs336_basics.naive_tokenizer.NaiveTokenizer(result.vocab, result.merges, result.special_tokens)


def store_tokenizer(state_file: str | os.PathLike, tokenizer: cs336_basics.naive_tokenizer.NaiveTokenizer) -> None:
    with open(state_file, "w") as f:
        # attri = {
        #     "vocab": tokenizer.vocab,
        #     "merges": tokenizer.merges,
        #     "special_tokens": tokenizer.special_tokens
        # }
        tokenizer_json_data = naive_tokenizer.JSONEncodeData.from_tokenizer(tokenizer)
        json_str = msgspec.json.encode(tokenizer_json_data).decode("utf-8")
        f.write(json_str)
        f.flush()


def load_tokenizer(file_path: str) -> cs336_basics.naive_tokenizer.NaiveTokenizer:
    return new_from_state_file(file_path)


def init_tokenizer(training_data_file: str, vocab_size: int,
                   special_tokens: list[str], ) -> cs336_basics.naive_tokenizer.NaiveTokenizer:
    tokenizer_file = "{}:{}:{}.json".format(training_data_file, vocab_size, special_tokens)
    if os.path.exists(tokenizer_file):
        return load_tokenizer(tokenizer_file)
    # training the bpe
    vocab, merges = ran_train_bpe.run_train_bpe(training_data_file, vocab_size, special_tokens)
    tokenizer = naive_tokenizer.NaiveTokenizer(vocab, merges, special_tokens)
    store_tokenizer(tokenizer_file, tokenizer)
    return tokenizer


def tokenize_file_and_persistent(file_path: str, split_special_token: str, tokenizer: naive_tokenizer.NaiveTokenizer):
    target_file_path = "{}.{}".format(file_path, TOKENIZED_FILE_SUFFIX)
    if os.path.exists(target_file_path):
        return target_file_path
    with open(target_file_path, "wb") as target_file:
        with open(file_path, "rb") as f:
            text = bytearray()
            for line in f:
                text.extend(line)
                if split_special_token in line.decode("utf-8"):
                    tokens = tokenizer.encode(text.decode("utf-8"))
                    arr = np.array(tokens, dtype=np.uint16)
                    arr.tofile(target_file)
                    text.clear()
        if text:
            tokens = tokenizer.encode(text.decode("utf-8"))
            np.asarray(tokens, dtype=np.uint16).tofile(target_file)
    return target_file_path


def get_batch(dataset_file, batch_size, seq_len, device: str | None = None):
    data = np.memmap(dataset_file, dtype=np.uint16, mode="r")
    ix = np.random.randint(0, len(data) - seq_len - 1, size=batch_size)

    x = np.stack([data[i: i + seq_len] for i in ix]).astype(np.int64)
    y = np.stack([data[i + 1: i + 1 + seq_len] for i in ix]).astype(np.int64)

    x, y = torch.from_numpy(x), torch.from_numpy(y)
    x = x.to(device)
    y = y.to(device)
    # if device is not None and device.startswith("mps"):
    #     x = x.pin_memory().to(device, non_blocking=True)
    #     y = y.pin_memory().to(device, non_blocking=True)
    return x, y


def train(dataset_file: str, module: torch.nn.Module,
          optimizer: torch.optim.Optimizer, batch_size: int, seq_len: int, total_steps: int,
          device: str | None = None) -> None:
    for step in range(total_steps):
        batch, label = get_batch(dataset_file, batch_size, seq_len, device=device)
        label = label.to(device)
        logits, loss = module(batch, targets=label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(step, loss.item())
