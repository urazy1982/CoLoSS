"""Subset utilities. A composition of n adapters is an int bitmask in [0, 2**n)."""
from __future__ import annotations

from itertools import combinations

import numpy as np


def popcount(s: int) -> int:
    return bin(s).count("1")


def members(s: int, n: int) -> list[int]:
    return [i for i in range(n) if s >> i & 1]


def mask(items) -> int:
    m = 0
    for i in items:
        m |= 1 << i
    return m


def all_subsets(n: int) -> np.ndarray:
    return np.arange(2 ** n, dtype=np.int64)


def subsets_of_size(n: int, k: int) -> list[int]:
    return [mask(c) for c in combinations(range(n), k)]


def interaction_terms(n: int, max_degree: int) -> list[int]:
    """All index sets T with |T| <= max_degree, as bitmasks, ordered by degree."""
    terms = []
    for k in range(max_degree + 1):
        terms.extend(subsets_of_size(n, k))
    return terms


def bit_matrix(n: int) -> np.ndarray:
    """(2**n, n) 0/1 matrix: row s is the inclusion vector of subset s."""
    s = all_subsets(n)[:, None]
    return ((s >> np.arange(n)[None, :]) & 1).astype(np.int8)


def mobius_features(subsets: np.ndarray, terms: list[int]) -> np.ndarray:
    """AND basis: phi_T(S) = 1[T subset of S]. A planted conjunction 'T within S' is 1-sparse here."""
    S = np.asarray(subsets, dtype=np.int64)[:, None]
    T = np.asarray(terms, dtype=np.int64)[None, :]
    return ((S & T) == T).astype(np.float64)


def walsh_features(subsets: np.ndarray, terms: list[int]) -> np.ndarray:
    """Walsh-Hadamard basis: chi_T(S) = (-1)^{|T cap S|}."""
    S = np.asarray(subsets, dtype=np.int64)[:, None]
    T = np.asarray(terms, dtype=np.int64)[None, :]
    inter = S & T
    parity = np.zeros(inter.shape, dtype=np.int64)
    x = inter.copy()
    while np.any(x):
        parity ^= x & 1
        x >>= 1
    return 1.0 - 2.0 * parity
