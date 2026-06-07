"""Cache-key helpers for precomputed isotope FT tables."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence


@dataclass(frozen=True)
class TableCacheKey:
    """Identity for a precomputed centered log/phase table."""

    elements: tuple[str, ...]
    dm: float
    n_fft: int
    isotope_data_version: str

    def as_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["elements"] = list(self.elements)
        return result


def make_table_cache_key(
    *,
    elements: Sequence[str],
    dm: float,
    n_fft: int,
    isotope_data_version: str,
) -> TableCacheKey:
    """Build a normalized cache key for table reuse."""

    return TableCacheKey(
        elements=tuple(elements),
        dm=float(dm),
        n_fft=int(n_fft),
        isotope_data_version=str(isotope_data_version),
    )
