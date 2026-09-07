from __future__ import annotations

import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import regex as re
import heapq
from .pretokenization_example import find_chunk_boundaries

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

type Pair = tuple[int, int]
type PairPosition = tuple[Node, Node]
type PreTokenCounter = Counter[bytes]


def process_chunk(
    file_path,
    start,
    end,
    special_tokens: list[str],
):
    counter: PreTokenCounter = Counter()

    with open(file_path, "rb") as f:
        f.seek(start)
        data = f.read(end - start).decode("utf-8")

    if special_tokens:
        special_pattern = "|".join(
            re.escape(token)
            for token in sorted(
                special_tokens,
                key=len,
                reverse=True,
            )
        )

        # Remove special tokens from normal training text
        text_chunks = re.split(special_pattern, data)
    else:
        text_chunks = [data]

    for text in text_chunks:
        for pre_token in re.findall(PAT, text):
            counter[pre_token.encode("utf-8")] += 1

    return counter

def pre_token_count(
        input_path: str | os.PathLike,
        split_special_token: list[str],
) -> PreTokenCounter:
    pre_token_counter: PreTokenCounter = Counter()
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, 1000, split_special_token)
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(
                process_chunk,
                input_path,
                start,
                end,
                split_special_token,
            )
            for start, end in zip(boundaries[:-1], boundaries[1:])
        ]
        results = [f.result() for f in futures]
    for result in results:
        for key, value in result.items():
            pre_token_counter[key] += value
    return pre_token_counter


def encode_utf8_integer(text: str) -> list[int]:
    return list(text.encode("utf-8"))


def decode_pair(
        pair: Pair,
        vocab: dict[int, bytes],
) -> tuple[bytes, tuple[bytes, bytes]]:
    first = vocab[pair[0]]
    second = vocab[pair[1]]

    return first + second, (first, second)


def split_chunks(
        # content: str,
        counter: PreTokenCounter,
        special_tokens: list[str],
) -> list[Node]:
    result: list[Node] = []
    for pre_token_with_counter in counter.items():
        pre_token_int_slice = pre_token_with_counter[0]
        count = pre_token_with_counter[1]
        token_ids = [int(token) for token in pre_token_int_slice]
        node = Node(0, token_ids[0], multiplier=count)
        root = node
        idx = 1
        while idx < len(pre_token_int_slice):
            next_node = Node(idx, pre_token_int_slice[idx], prev_node=node, multiplier=count)
            node.next(next_node)
            node = node.next_node
            idx += 1
        result.append(root)
    return result


def init_stats_chunked(
        ids_chunks: list[Node],
) -> tuple[dict[Pair, int], dict[Pair, list[PairPosition]]]:
    stats: dict[Pair, int] = {}
    result_positions: dict[Pair, list[PairPosition]] = {}
    for node in ids_chunks:
        while node is not None and node.next_node is not None:
            pair = (node.token_id, node.next_node.token_id)
            stats[pair] = stats.get(pair, 0) + node.multiplier
            if pair not in result_positions:
                result_positions[pair] = list()
            result_positions.get(pair).append((node, node.next_node))
            node = node.next_node
    return stats, result_positions


# 重新计算PQ 中的top,看里面有多少失效的node
def pop_top_pair(pq: list[HeapItem], vocab, result_positions: dict[Pair, list[PairPosition]]) -> Pair:
    # (count, _, _, pair) = heapq.heappop(pq)

    heap_item = heapq.heappop(pq)
    pair = heap_item.pair
    while len(pq) > 0:
        pair_position_set = result_positions[pair]
        new_set = list()
        new_count = 0
        for pair_position in pair_position_set:
            if pair_position[0].active and pair_position[1].active:
                new_set.append(pair_position)
                new_count += pair_position[0].multiplier
        # heapq.heappush(pq, (0 - new_count, vocab[pair[0]], vocab[pair[1]], pair))
        heapq.heappush(pq, HeapItem(new_count, pair, vocab))
        result_positions[pair] = new_set
        # _, _, _, new_pair = heapq.heappop(pq)
        heap_item = heapq.heappop(pq)
        if heap_item.pair[0] == pair[0] and heap_item.pair[1] == pair[1]:
            break
        pair = heap_item.pair
    return pair


# merge 的过程中会出现新的 pair，新的pair 不仅仅要 一个个 count,还有记录新的pair的positions re_stats,新的一批pair
def merge_optimised(
        node_pairs: list[PairPosition],
        new_id: int
):
    stats: dict[Pair, int] = {}
    # pair --> all pair node list
    pair_positions: dict[Pair, list[PairPosition]] = {}
    for node_pair in node_pairs:
        first, second = node_pair
        if not first.active or not second.active:
            continue
        first.deactivate()
        second.deactivate()

        prev_node = first.prev_node
        next_node = second.next_node
        new_node = Node(
            idx=first.idx,
            token_id=new_id,
            next_node=next_node,
            prev_node=prev_node,
            multiplier=first.multiplier,
        )
        if first.prev_node is not None:
            first.prev_node.next_node = new_node
        if second.next_node is not None:
            second.next_node.prev_node = new_node
        if new_node.prev_node is not None:
            new_pair = (new_node.prev_node.token_id, new_id)
            if new_pair not in pair_positions:
                pair_positions[new_pair] = list()
            pair_positions[new_pair].append((new_node.prev_node, new_node))
            stats[new_pair] = stats.get(new_pair, 0) + new_node.multiplier
        if new_node.next_node is not None:
            new_pair = (new_id, new_node.next_node.token_id)
            if new_pair not in pair_positions:
                pair_positions[new_pair] = list()
            pair_positions[new_pair].append((new_node, new_node.next_node))
            stats[new_pair] = stats.get(new_pair, 0) + new_node.multiplier
    return stats, pair_positions


def merge(
        node: Node,
        pair: Pair,
        new_id: int,
) -> Node:
    root = node

    while node is not None and node.next_node is not None:
        if (
                node.token_id == pair[0]
                and node.next_node.token_id == pair[1]
        ):
            prev_node = node.prev_node
            next_node = node.next_node.next_node

            new_node = Node(
                idx=node.idx,
                token_id=new_id,
                next_node=next_node,
                prev_node=prev_node,
                multiplier=node.multiplier,
            )

            if prev_node is None:
                root = new_node
            else:
                prev_node.next_node = new_node

            if next_node is not None:
                next_node.prev_node = new_node

            # Continue after the newly merged token.
            node = new_node.next_node
        else:
            node = node.next_node

    return root


class Node:
    def __init__(self, idx: int, token_id: int, next_node: Node | None = None, prev_node: Node | None = None,
                 multiplier: int = 1):
        self.idx: int = idx
        self.token_id: int = token_id
        self.next_node = next_node
        self.prev_node = prev_node
        self.active = True
        self.multiplier = multiplier

    def prev(self, prev: Node):
        self.prev_node = prev

    def next(self, next_node: Node):
        self.next_node = next_node

    def deactivate(self):
        self.active = False


class HeapItem:
    def __init__(self, count, pair, vocab):
        self.count = count
        self.bytes_key = (vocab[pair[0]], vocab[pair[1]])
        self.pair = pair

    def __lt__(self, other):
        if self.count != other.count:
            return self.count > other.count  # count 大的先出
        return self.bytes_key > other.bytes_key  # 打平时 bytes 大的先出


def run_train_bpe(
        input_path: str | os.PathLike,
        vocab_size: int,
        special_tokens: list[str],
        **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    if vocab_size < 256 + len(special_tokens):
        raise ValueError(
            f"vocab_size must be at least "
            f"{256 + len(special_tokens)}, got {vocab_size}"
        )

    vocab: dict[int, bytes] = {}
    merges: list[tuple[bytes, bytes]] = []

    for token_id in range(256):
        vocab[token_id] = bytes([token_id])
    result_counter = pre_token_count(input_path, special_tokens)
    ids_chunked = split_chunks(
        result_counter,
        special_tokens,
    )

    num_merges = vocab_size - 256 - len(special_tokens)

    next_token_id = 256

    pq: list[HeapItem] = []
    stats, positions = init_stats_chunked(ids_chunked)
    # No more adjacent pairs exist.
    if not stats or len(stats) == 0:
        return vocab, merges
    for stat in stats.items():
        heapq.heappush(pq, HeapItem(stat[1], stat[0], vocab))
    for _ in range(num_merges):
        target_pair = pop_top_pair(pq, vocab, positions)
        token_bytes, merge_bytes = decode_pair(
            target_pair,
            vocab,
        )

        # Add merged token to vocabulary.
        vocab[next_token_id] = token_bytes
        merges.append(merge_bytes)
        new_stats, new_pair_positions = merge_optimised(positions[target_pair], next_token_id)
        for new_stat in new_stats.items():
            stats[new_stat[0]] = new_stat[1]
            heapq.heappush(pq, HeapItem(new_stat[1], new_stat[0], vocab))
            # heapq.heappush(pq, (0 - , vocab[new_stat[0][0]], vocab[new_stat[0][1]], new_stat[0]))
        for new_pair_position in new_pair_positions.items():
            positions[new_pair_position[0]] = new_pair_position[1]

        next_token_id += 1

    # ------------------------------------------------------------------
    # 5. Add special tokens
    # ------------------------------------------------------------------

    for special_token in special_tokens:
        vocab[next_token_id] = special_token.encode("utf-8")
        next_token_id += 1

    return vocab, merges
