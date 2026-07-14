"""Utilities for reproducible scientific experiments."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import random
from typing import Any

import numpy as np


def set_global_seed(seed: int, *, include_torch: bool = False) -> dict[str, Any]:
    """Seed Python, NumPy and optionally PyTorch.

    The returned dictionary is intended to be persisted with run artifacts.
    """
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    audit: dict[str, Any] = {
        "seed": seed,
        "python_random": True,
        "numpy": True,
        "torch": False,
    }

    if include_torch and importlib.util.find_spec("torch") is not None:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
        audit["torch"] = True
        audit["torch_cuda"] = bool(torch.cuda.is_available())

    return audit


def derive_seed(base_seed: int, *parts: object) -> int:
    """Derive a stable 32-bit seed from a base seed and arbitrary labels."""
    if not isinstance(base_seed, int) or isinstance(base_seed, bool) or base_seed < 0:
        raise ValueError("base_seed must be a non-negative integer")

    payload = "|".join([str(base_seed), *(str(part) for part in parts)])
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False) % (2**32)
