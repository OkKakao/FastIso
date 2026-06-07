"""Odd-length inverse real FFT helpers.

The isotope-profile engine will usually work from a positive-frequency template.
Using an odd output length avoids the special real-valued Nyquist bin that exists
for even-length real FFTs, while still allowing fast FFT execution when the
length has only small prime factors.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from operator import index
from typing import Iterable

import numpy as np
from numpy.lib.array_utils import normalize_axis_index
from scipy.fft import irfft, rfftfreq
from scipy.signal import czt

DEFAULT_ODD_PRIMES: tuple[int, ...] = (3, 5, 7)


@dataclass(frozen=True)
class FastOddFFTPlan:
    """Description of an odd-length real inverse FFT plan."""

    min_len: int
    n_fft: int
    n_positive: int
    factorization: dict[int, int]


def factorize(n: int) -> dict[int, int]:
    """Return the prime factorization of a positive integer."""

    n = _as_positive_int(n, "n")
    factors: dict[int, int] = {}
    divisor = 2
    while divisor * divisor <= n:
        while n % divisor == 0:
            factors[divisor] = factors.get(divisor, 0) + 1
            n //= divisor
        divisor += 1 if divisor == 2 else 2
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors


def positive_frequency_count(n_fft: int) -> int:
    """Return the number of bins in an `rfft`/`irfft` spectrum."""

    n_fft = _as_positive_int(n_fft, "n_fft")
    return n_fft // 2 + 1


def is_odd_smooth_length(
    n_fft: int,
    *,
    allowed_primes: Iterable[int] = DEFAULT_ODD_PRIMES,
) -> bool:
    """Return whether `n_fft` is odd and factors only into `allowed_primes`."""

    n_fft = _as_positive_int(n_fft, "n_fft")
    if n_fft % 2 == 0:
        return False
    primes = _validate_odd_primes(allowed_primes)
    return all(prime in primes for prime in factorize(n_fft))


def suggest_fast_odd_len(
    min_len: int,
    *,
    allowed_primes: Iterable[int] = DEFAULT_ODD_PRIMES,
    max_overshoot: float = 1.25,
) -> int:
    """Choose the smallest odd smooth FFT length at least `min_len`.

    The default prime set `(3, 5, 7)` intentionally avoids lengths such as
    `32769 = 3^2 * 11 * 331`, which are odd but slow because of the large prime
    factor. For example, `suggest_fast_odd_len(32768)` returns `32805`.
    """

    min_len = _as_positive_int(min_len, "min_len")
    primes = _validate_odd_primes(allowed_primes)
    if min_len == 1:
        return 1

    lower = min_len if min_len % 2 else min_len + 1
    limit = max(lower, ceil(min_len * max_overshoot))
    while True:
        candidates = _smooth_lengths_up_to(limit, primes)
        valid = [n for n in candidates if n >= lower]
        if valid:
            return min(valid)
        limit *= 2


def plan_fast_odd_irfft(
    min_len: int,
    *,
    allowed_primes: Iterable[int] = DEFAULT_ODD_PRIMES,
    max_overshoot: float = 1.25,
) -> FastOddFFTPlan:
    """Build a small immutable plan object for a fast odd-length `irfft`."""

    n_fft = suggest_fast_odd_len(
        min_len,
        allowed_primes=allowed_primes,
        max_overshoot=max_overshoot,
    )
    return FastOddFFTPlan(
        min_len=min_len,
        n_fft=n_fft,
        n_positive=positive_frequency_count(n_fft),
        factorization=factorize(n_fft),
    )


def rfftfreq_odd(n_fft: int, d: float = 1.0) -> np.ndarray:
    """Return positive FFT frequencies for an odd-length real transform."""

    n_fft = _as_positive_int(n_fft, "n_fft")
    if n_fft % 2 == 0:
        raise ValueError(f"n_fft must be odd, got {n_fft}")
    return rfftfreq(n_fft, d=d)


def fast_odd_irfft(
    spectrum: np.ndarray,
    *,
    n: int | None = None,
    min_len: int | None = None,
    axis: int = -1,
    norm: str | None = None,
    workers: int | None = None,
) -> np.ndarray:
    """Reconstruct a real signal from a positive-frequency spectrum.

    Parameters
    ----------
    spectrum:
        Complex positive-frequency bins in SciPy/NumPy `rfft` ordering.
    n:
        Explicit odd output length. If omitted, `min_len` is converted to a fast
        odd length. If both are omitted, the odd length implied by the spectrum
        size is used: `n = 2 * n_positive - 1`.
    min_len:
        Minimum requested output length when `n` is not supplied.
    axis:
        Transform axis.
    norm, workers:
        Passed through to `scipy.fft.irfft`.
    """

    spectrum = np.asarray(spectrum)
    n_fft = _resolve_n_fft(spectrum, n=n, min_len=min_len, axis=axis)
    if n_fft % 2 == 0:
        raise ValueError(f"fast_odd_irfft requires an odd output length, got {n_fft}")
    return irfft(spectrum, n=n_fft, axis=axis, norm=norm, workers=workers)


def czt_irfft_window(
    spectrum: np.ndarray,
    *,
    n: int | None = None,
    min_len: int | None = None,
    start: float,
    step: float,
    m: int,
    axis: int = -1,
    trim_zeros: bool = False,
) -> np.ndarray:
    """Evaluate an odd-length inverse real FFT on a regular sample window.

    ``start`` and ``step`` are expressed in sample-index units of the implicit
    odd-length inverse transform. Integer positions match ``fast_odd_irfft``;
    fractional positions evaluate the same periodic, band-limited inverse DFT
    interpolation without materializing the whole output profile.
    """

    spectrum = np.asarray(spectrum)
    n_fft = _resolve_n_fft(spectrum, n=n, min_len=min_len, axis=axis)
    if n_fft % 2 == 0:
        raise ValueError(f"czt_irfft_window requires an odd output length, got {n_fft}")
    m = _as_positive_int(m, "m")
    axis = normalize_axis_index(axis, spectrum.ndim)
    expected_positive = positive_frequency_count(n_fft)
    if spectrum.shape[axis] != expected_positive:
        raise ValueError(
            "spectrum positive-frequency length does not match n: "
            f"expected {expected_positive}, got {spectrum.shape[axis]}"
        )
    if trim_zeros:
        spectrum = _trim_trailing_zeros(spectrum, axis=axis)

    # scipy.signal.czt evaluates sum x[k] a**(-k) w**(j*k). With these
    # parameters that becomes the positive-frequency part of the inverse DFT at
    # t_j = start + j * step.
    a = np.exp(-2j * np.pi * float(start) / n_fft)
    w = np.exp(2j * np.pi * float(step) / n_fft)
    positive_sum = czt(spectrum, m=m, w=w, a=a, axis=axis)

    dc = np.take(spectrum.real, indices=0, axis=axis)
    dc = np.expand_dims(dc, axis=axis)
    return (2.0 * positive_sum.real - dc) / n_fft


def _resolve_n_fft(
    spectrum: np.ndarray,
    *,
    n: int | None,
    min_len: int | None,
    axis: int,
) -> int:
    if n is not None and min_len is not None:
        raise ValueError("Pass either n or min_len, not both")
    if n is not None:
        return _as_positive_int(n, "n")
    if min_len is not None:
        return suggest_fast_odd_len(min_len)

    axis = normalize_axis_index(axis, spectrum.ndim)
    n_positive = spectrum.shape[axis]
    if n_positive < 1:
        raise ValueError("spectrum must contain at least one positive-frequency bin")
    return 2 * n_positive - 1


def _trim_trailing_zeros(spectrum: np.ndarray, *, axis: int) -> np.ndarray:
    reduction_axes = tuple(idx for idx in range(spectrum.ndim) if idx != axis)
    if reduction_axes:
        nonzero = np.any(spectrum != 0.0, axis=reduction_axes)
    else:
        nonzero = spectrum != 0.0
    indices = np.flatnonzero(nonzero)
    if indices.size == 0:
        return np.take(spectrum, [0], axis=axis)
    last = int(indices[-1]) + 1
    return np.take(spectrum, np.arange(last), axis=axis)


def _smooth_lengths_up_to(limit: int, primes: tuple[int, ...]) -> list[int]:
    values: list[int] = []

    def visit(index: int, current: int) -> None:
        if index == len(primes):
            values.append(current)
            return
        value = current
        prime = primes[index]
        while value <= limit:
            visit(index + 1, value)
            value *= prime

    visit(0, 1)
    return values


def _validate_odd_primes(primes: Iterable[int]) -> tuple[int, ...]:
    unique = tuple(sorted(set(primes)))
    if not unique:
        raise ValueError("allowed_primes must not be empty")
    for prime in unique:
        prime = _as_positive_int(prime, "allowed prime")
        if prime == 1 or prime % 2 == 0 or factorize(prime) != {prime: 1}:
            raise ValueError(f"allowed_primes must contain odd primes, got {prime}")
    return unique


def _as_positive_int(value: int, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        result = index(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if result < 1:
        raise ValueError(f"{name} must be positive")
    return result
