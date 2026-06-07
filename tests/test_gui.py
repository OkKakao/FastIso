import numpy as np
import pytest

pytest.importorskip("tkinter")

from fastiso.cli import simulate_profiles
from fastiso.gui import _profile_plot_points, _profile_preview_indices


def test_gui_preview_keeps_sulfur_isotope_peaks():
    mass_axis, intensity = _sulfur_profile()

    indices = _profile_preview_indices(mass_axis, intensity, max_rows=10)
    selected_masses = mass_axis[indices]

    for target in _SULFUR_ISOTOPE_MASSES:
        assert np.any(np.abs(selected_masses - target) < 0.001)


def test_gui_plot_downsampling_keeps_sulfur_isotope_peaks():
    mass_axis, intensity = _sulfur_profile()

    plot_mass, _plot_intensity = _profile_plot_points(
        mass_axis,
        intensity,
        max_points=1500,
    )

    assert plot_mass.size < mass_axis.size
    for target in _SULFUR_ISOTOPE_MASSES:
        assert np.any(np.abs(plot_mass - target) < 0.001)


_SULFUR_ISOTOPE_MASSES = (
    31.9720711744,
    32.9714589098,
    33.967867004,
    35.96708071,
)


def _sulfur_profile() -> tuple[np.ndarray, np.ndarray]:
    result = simulate_profiles(
        ["S"],
        elements=["S"],
        window_mode="auto",
        auto_grid=True,
        min_fft_len=255,
        method="log_pruned",
        storage_mode="research",
    )
    return np.asarray(result["mass_axis"][0]), np.asarray(result["intensity"][0])
