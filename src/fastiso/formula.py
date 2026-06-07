"""Formula parsing utilities."""

from __future__ import annotations

from collections import defaultdict
import re
from typing import Mapping, Sequence

import numpy as np

_TOKEN_RE = re.compile(r"[A-Z][a-z]?|\(|\)|\d+")


def parse_formula(formula: str) -> dict[str, int]:
    """Parse a neutral chemical formula into integer element counts.

    The parser supports nested parentheses, for example ``C6H5(CH3)`` and
    ``(CH3)2O``. Charges, adduct notation, isotope labels, and decimal counts are
    intentionally outside this first prototype.
    """

    if not isinstance(formula, str) or not formula:
        raise ValueError("formula must be a non-empty string")

    tokens = _tokenize(formula)
    counts, position = _parse_group(tokens, 0)
    if position != len(tokens):
        raise ValueError(f"unexpected token {tokens[position]!r} in formula {formula!r}")
    return dict(counts)


def counts_matrix(
    formulas: Sequence[str | Mapping[str, int]],
    elements: Sequence[str],
) -> np.ndarray:
    """Convert formulas into an ``(n_formulas, n_elements)`` count matrix."""

    element_index = {element: idx for idx, element in enumerate(elements)}
    matrix = np.zeros((len(formulas), len(elements)), dtype=np.int64)
    for row_idx, formula in enumerate(formulas):
        parsed = parse_formula(formula) if isinstance(formula, str) else dict(formula)
        for element, count in parsed.items():
            if element not in element_index:
                raise ValueError(f"element {element!r} is not available in this table")
            if count < 0:
                raise ValueError(f"negative count for element {element!r}: {count}")
            matrix[row_idx, element_index[element]] = int(count)
    return matrix


def _tokenize(formula: str) -> list[str]:
    tokens: list[str] = []
    position = 0
    for match in _TOKEN_RE.finditer(formula):
        if match.start() != position:
            raise ValueError(f"unsupported formula syntax near {formula[position:]!r}")
        tokens.append(match.group(0))
        position = match.end()
    if position != len(formula):
        raise ValueError(f"unsupported formula syntax near {formula[position:]!r}")
    return tokens


def _parse_group(tokens: Sequence[str], position: int) -> tuple[dict[str, int], int]:
    counts: defaultdict[str, int] = defaultdict(int)
    while position < len(tokens):
        token = tokens[position]
        if token == ")":
            break
        if token == "(":
            child, position = _parse_group(tokens, position + 1)
            if position >= len(tokens) or tokens[position] != ")":
                raise ValueError("unclosed parenthesis in formula")
            position += 1
            multiplier, position = _parse_multiplier(tokens, position)
            for element, count in child.items():
                counts[element] += count * multiplier
            continue
        if token.isdigit():
            raise ValueError(f"unexpected multiplier {token!r}")

        element = token
        position += 1
        multiplier, position = _parse_multiplier(tokens, position)
        counts[element] += multiplier
    return dict(counts), position


def _parse_multiplier(tokens: Sequence[str], position: int) -> tuple[int, int]:
    if position < len(tokens) and tokens[position].isdigit():
        multiplier = int(tokens[position])
        if multiplier < 1:
            raise ValueError("formula multipliers must be positive")
        return multiplier, position + 1
    return 1, position

