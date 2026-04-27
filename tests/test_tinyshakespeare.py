import numpy as np
import pytest

from jacobigrad.data.tinyshakespeare import CharTokenizer, get_batch


def test_tokenizer_vocab_is_sorted_unique():
    tok = CharTokenizer.from_text("banana")
    assert tok.vocab_size == 3
    assert tok.itos == ("a", "b", "n")
    assert tok.stoi == {"a": 0, "b": 1, "n": 2}


def test_tokenizer_roundtrip():
    text = "To be or not to be, that is the question."
    tok = CharTokenizer.from_text(text)
    ids = tok.encode(text)
    assert ids.dtype == np.int64
    assert ids.shape == (len(text),)
    assert tok.decode(ids) == text


def test_tokenizer_unknown_char_raises():
    tok = CharTokenizer.from_text("abc")
    with pytest.raises(KeyError):
        tok.encode("abcd")


def test_get_batch_shape():
    rng = np.random.default_rng(0)
    data = np.arange(100, dtype=np.int64)
    x, y = get_batch(data, batch_size=4, block_size=8, rng=rng)
    assert x.shape == (4, 8)
    assert y.shape == (4, 8)
    assert x.dtype == np.int64
    assert y.dtype == np.int64


def test_get_batch_shift_invariant():
    # y is x shifted by one within each window, so y[:, :-1] == x[:, 1:].
    # This is the contract that makes (x, y) a valid next-char training pair.
    rng = np.random.default_rng(0)
    data = np.arange(100, dtype=np.int64)
    x, y = get_batch(data, batch_size=4, block_size=8, rng=rng)
    assert np.array_equal(y[:, :-1], x[:, 1:])


def test_get_batch_deterministic_with_seed():
    data = np.arange(100, dtype=np.int64)
    x1, y1 = get_batch(data, 4, 8, np.random.default_rng(42))
    x2, y2 = get_batch(data, 4, 8, np.random.default_rng(42))
    assert np.array_equal(x1, x2)
    assert np.array_equal(y1, y2)


def test_get_batch_starts_within_bounds():
    # Stress test: with block_size = len(data) - 1, the only legal start is 0.
    rng = np.random.default_rng(0)
    data = np.arange(10, dtype=np.int64)
    x, y = get_batch(data, batch_size=5, block_size=9, rng=rng)
    assert np.all(x == data[:9])
    assert np.all(y == data[1:10])


def test_get_batch_too_short_raises():
    rng = np.random.default_rng(0)
    data = np.arange(5, dtype=np.int64)
    with pytest.raises(ValueError, match="too short"):
        get_batch(data, batch_size=2, block_size=8, rng=rng)
