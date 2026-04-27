"""Tiny Shakespeare dataset: char-level tokenizer, loader, batched sampler.

The corpus is the concatenation of selected Shakespeare plays — about 1.1 MB
of ASCII text and ~65 unique characters. It is the canonical toy benchmark
for character-level language modeling, used by Karpathy's char-rnn and
nanoGPT.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np

# Provenance: this is the input.txt shipped with Karpathy's char-rnn repo.
TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)
DEFAULT_CACHE = Path("data") / "tiny_shakespeare.txt"


class CharTokenizer:
    """Deterministic character-level tokenizer.

    The vocabulary is the sorted set of unique characters in the corpus the
    tokenizer is built from. IDs are assigned 0..V-1 in that order, so token
    IDs are stable across runs as long as the source corpus is the same.
    """

    def __init__(self, stoi: dict[str, int], itos: tuple[str, ...]):
        self.stoi = stoi
        self.itos = itos

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        chars = sorted(set(text))
        stoi = {c: i for i, c in enumerate(chars)}
        itos = tuple(chars)
        return cls(stoi=stoi, itos=itos)

    def encode(self, text: str) -> np.ndarray:
        return np.array([self.stoi[c] for c in text], dtype=np.int64)

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)


def download_tinyshakespeare(path: Path = DEFAULT_CACHE) -> Path:
    """Download Tiny Shakespeare to ``path`` if not cached. Idempotent."""
    path = Path(path)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(TINY_SHAKESPEARE_URL) as response:
        path.write_bytes(response.read())
    return path


def load_tinyshakespeare(
    cache_path: Path = DEFAULT_CACHE,
    val_fraction: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, CharTokenizer]:
    """Load and tokenize the corpus; return ``(train_ids, val_ids, tokenizer)``.

    The split is by *position* (last ``val_fraction`` of the encoded corpus is
    val). A position-based split preserves local statistics within each split;
    random char-level shuffling would leak context across the boundary and
    inflate val performance in a way that doesn't reflect generalization.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1); got {val_fraction}")
    path = download_tinyshakespeare(cache_path)
    text = path.read_text(encoding="utf-8")
    tokenizer = CharTokenizer.from_text(text)
    ids = tokenizer.encode(text)
    n = len(ids)
    n_val = int(n * val_fraction)
    train_ids = ids[: n - n_val]
    val_ids = ids[n - n_val :]
    return train_ids, val_ids, tokenizer


def get_batch(
    split: np.ndarray,
    batch_size: int,
    block_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a batch of (input, target) windows from a token array.

    Returns ``x, y`` of shape ``(batch_size, block_size)``. ``y`` is ``x``
    shifted by one position, so ``y[i, t]`` is the next-character target for
    ``x[i, t]``. The caller passes an ``np.random.Generator`` so sampling is
    reproducible across runs.
    """
    if len(split) < block_size + 1:
        raise ValueError(
            f"split too short ({len(split)}) for block_size={block_size}; "
            f"need >= {block_size + 1}"
        )
    # Valid start s satisfies s + block_size + 1 <= len(split), i.e.
    # s in [0, len(split) - block_size - 1]. rng.integers' high is exclusive.
    max_start_exclusive = len(split) - block_size
    starts = rng.integers(0, max_start_exclusive, size=batch_size)
    x = np.stack([split[s : s + block_size] for s in starts])
    y = np.stack([split[s + 1 : s + block_size + 1] for s in starts])
    return x, y
