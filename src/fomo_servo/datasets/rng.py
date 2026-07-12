"""Stable epoch-aware RNG helpers for online augmentation."""

from __future__ import annotations

import hashlib
import struct

import numpy as np


def stable_sample_seed(base_seed: int, epoch: int, sample_index: int) -> int:
    """Derive a stable unsigned 64-bit seed from base/epoch/index."""

    values = (base_seed, epoch, sample_index)
    for name, value in zip(("base_seed", "epoch", "sample_index"), values):
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**64:
            raise ValueError("{} must be an integer in [0, 2**64)".format(name))
    payload = struct.pack("<QQQ", base_seed, epoch, sample_index)
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def make_sample_rng(base_seed: int, epoch: int, sample_index: int) -> np.random.Generator:
    """Create the deterministic per-sample NumPy generator."""

    return np.random.default_rng(stable_sample_seed(base_seed, epoch, sample_index))


__all__ = ["make_sample_rng", "stable_sample_seed"]
