"""Fast FT-based isotope profile simulation utilities."""

from .fast_odd_irfft import (
    DEFAULT_ODD_PRIMES,
    FastOddFFTPlan,
    czt_irfft_window,
    factorize,
    fast_odd_irfft,
    is_odd_smooth_length,
    plan_fast_odd_irfft,
    positive_frequency_count,
    rfftfreq_odd,
    suggest_fast_odd_len,
)
from .formula import counts_matrix, parse_formula
from .cache import TableCacheKey, make_table_cache_key
from .isotopes import (
    ELEMENT_PRESETS,
    FormulaIsotopeComponents,
    IsotopePattern,
    IsotopeRegistry,
    default_isotope_patterns,
    isotope_data_version,
    load_isotope_patterns,
    load_isotope_registry,
    split_formula_isotope_components,
)
from .log_table import CenteredLogPhaseTable, has_cython_backend

__all__ = [
    "CenteredLogPhaseTable",
    "DEFAULT_ODD_PRIMES",
    "ELEMENT_PRESETS",
    "FastOddFFTPlan",
    "FormulaIsotopeComponents",
    "IsotopePattern",
    "IsotopeRegistry",
    "TableCacheKey",
    "counts_matrix",
    "czt_irfft_window",
    "default_isotope_patterns",
    "factorize",
    "fast_odd_irfft",
    "has_cython_backend",
    "is_odd_smooth_length",
    "isotope_data_version",
    "load_isotope_patterns",
    "load_isotope_registry",
    "make_table_cache_key",
    "parse_formula",
    "plan_fast_odd_irfft",
    "positive_frequency_count",
    "rfftfreq_odd",
    "suggest_fast_odd_len",
    "split_formula_isotope_components",
]
