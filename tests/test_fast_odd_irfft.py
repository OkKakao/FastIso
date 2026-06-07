import numpy as np
import pytest
from scipy.fft import rfft

from fastiso.fast_odd_irfft import (
    czt_irfft_window,
    factorize,
    fast_odd_irfft,
    is_odd_smooth_length,
    plan_fast_odd_irfft,
    positive_frequency_count,
    rfftfreq_odd,
    suggest_fast_odd_len,
)


def test_suggest_fast_odd_len_matches_memo_example():
    n_fft = suggest_fast_odd_len(32768)

    assert n_fft == 32805
    assert factorize(n_fft) == {3: 8, 5: 1}
    assert is_odd_smooth_length(n_fft)


def test_rejects_odd_length_with_large_prime_factor():
    assert not is_odd_smooth_length(32769)


def test_plan_reports_positive_frequency_count():
    plan = plan_fast_odd_irfft(1024)

    assert plan.n_fft % 2 == 1
    assert plan.n_positive == positive_frequency_count(plan.n_fft)
    assert set(plan.factorization).issubset({3, 5, 7})


def test_fast_odd_irfft_round_trips_scipy_rfft():
    rng = np.random.default_rng(123)
    signal = rng.normal(size=101)
    spectrum = rfft(signal)

    reconstructed = fast_odd_irfft(spectrum, n=101)

    np.testing.assert_allclose(reconstructed, signal, rtol=1e-12, atol=1e-12)


def test_fast_odd_irfft_infers_odd_length_from_spectrum():
    rng = np.random.default_rng(456)
    signal = rng.normal(size=129)
    spectrum = rfft(signal)

    reconstructed = fast_odd_irfft(spectrum)

    np.testing.assert_allclose(reconstructed, signal, rtol=1e-12, atol=1e-12)


def test_fast_odd_irfft_supports_batch_axis():
    rng = np.random.default_rng(789)
    signals = rng.normal(size=(4, 75))
    spectra = rfft(signals, axis=-1)

    reconstructed = fast_odd_irfft(spectra, n=75, axis=-1)

    np.testing.assert_allclose(reconstructed, signals, rtol=1e-12, atol=1e-12)


def test_czt_irfft_window_matches_integer_irfft_window():
    rng = np.random.default_rng(321)
    signal = rng.normal(size=101)
    spectrum = rfft(signal)

    window = czt_irfft_window(spectrum, n=101, start=17, step=1, m=25)

    np.testing.assert_allclose(window, signal[17:42], rtol=1e-12, atol=1e-12)


def test_czt_irfft_window_matches_direct_fractional_inverse_dft():
    rng = np.random.default_rng(654)
    signal = rng.normal(size=75)
    spectrum = rfft(signal)
    start = 3.25
    step = 0.5
    m = 18

    window = czt_irfft_window(spectrum, n=75, start=start, step=step, m=m)

    k = np.arange(spectrum.shape[0])
    t = start + step * np.arange(m)
    positive_sum = np.exp(2j * np.pi * np.outer(t, k) / 75) @ spectrum
    direct = (2.0 * positive_sum.real - spectrum[0].real) / 75
    np.testing.assert_allclose(window, direct, rtol=1e-12, atol=1e-12)


def test_czt_irfft_window_can_trim_trailing_zero_bins():
    rng = np.random.default_rng(987)
    signal = rng.normal(size=101)
    spectrum = rfft(signal)
    truncated = spectrum.copy()
    truncated[24:] = 0.0

    full_window = czt_irfft_window(
        truncated,
        n=101,
        start=5.5,
        step=0.75,
        m=30,
        trim_zeros=False,
    )
    trimmed_window = czt_irfft_window(
        truncated,
        n=101,
        start=5.5,
        step=0.75,
        m=30,
        trim_zeros=True,
    )

    np.testing.assert_allclose(trimmed_window, full_window, rtol=1e-12, atol=1e-12)


def test_rfftfreq_odd_has_no_nyquist_frequency():
    freqs = rfftfreq_odd(101, d=1.0)

    assert freqs[-1] == pytest.approx(50 / 101)
    assert freqs[-1] < 0.5


def test_fast_odd_irfft_rejects_even_output_length():
    with pytest.raises(ValueError, match="odd output length"):
        fast_odd_irfft(np.ones(5, dtype=np.complex128), n=8)
