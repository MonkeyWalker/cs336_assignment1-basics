import os
from typing import BinaryIO
import regex as re




def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    special_tokens: list[str],
) -> list[int]:
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    chunk_boundaries = [
        i * chunk_size
        for i in range(desired_num_chunks + 1)
    ]
    chunk_boundaries[-1] = file_size

    # Build regex only once
    special_tokens_bytes = [
        token.encode("utf-8")
        for token in special_tokens
    ]

    special_pattern = b"|".join(
        re.escape(token)
        for token in sorted(
            special_tokens_bytes,
            key=len,
            reverse=True,
        )
    )

    mini_chunk_size = 4096

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)

        while True:
            mini_chunk = file.read(mini_chunk_size)

            if not mini_chunk:
                chunk_boundaries[bi] = file_size
                break

            match = re.search(special_pattern, mini_chunk)

            if match:
                chunk_boundaries[bi] = initial_position + match.start()
                break

            initial_position += len(mini_chunk)

    return sorted(set(chunk_boundaries))




## Usage
# with open(..., "rb") as f:
#     num_processes = 4
#     boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
#
#     # The following is a serial implementation, but you can parallelize this
#     # by sending each start/end pair to a set of processes.
#     for start, end in zip(boundaries[:-1], boundaries[1:]):
#         f.seek(start)
#         chunk = f.read(end - start).decode("utf-8", errors="ignore")
#         # Run pre-tokenization on your chunk and store the counts for each pre-token
