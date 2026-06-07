import numpy as np
import pytest

pytest.importorskip("tkinter")

from fastiso.cli import simulate_profiles
from fastiso.gui import (
    _DEFAULT_CHARGE_STATE,
    _DEFAULT_PRESET,
    _axis_label_for_result,
    _dragged_x_center,
    _dragged_y_zoom,
    _format_peak_label,
    _label_peak_indices,
    _max_normalized_intensity,
    _resolving_power_from_sigma,
    _sigma_from_resolving_power,
    _single_formula_mean_mass,
    _profile_plot_points,
)


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


def test_gui_plot_uses_max_normalized_display_values():
    _mass_axis, intensity = _sulfur_profile()

    display = _max_normalized_intensity(intensity)

    assert np.max(display) == pytest.approx(1.0)
    assert np.sum(display) > 1.0


def test_gui_peak_labels_include_sulfur_isotope_peaks():
    mass_axis, intensity = _sulfur_profile()
    display = _max_normalized_intensity(intensity)

    indices = _label_peak_indices(mass_axis, display, max_labels=4)
    selected_masses = mass_axis[indices]

    for target in _SULFUR_ISOTOPE_MASSES:
        assert np.any(np.abs(selected_masses - target) < 0.001)


def test_gui_peak_label_uses_three_decimal_places():
    assert _format_peak_label(31.9720711744) == "31.972"


def test_gui_axis_drag_helpers_adjust_plot_view():
    center = _dragged_x_center(
        50.0,
        dx_pixels=100.0,
        plot_width=500.0,
        full_x_min=0.0,
        full_x_max=100.0,
        visible_width=20.0,
    )

    assert center == pytest.approx(46.0)
    assert _dragged_y_zoom(1.0, dy_pixels=-120.0) == pytest.approx(2.0)
    assert _dragged_y_zoom(1.0, dy_pixels=120.0) == pytest.approx(0.5)


def test_gui_defaults_cover_full_isotope_table():
    assert _DEFAULT_PRESET == "full"
    assert _DEFAULT_CHARGE_STATE == "0"
    assert _single_formula_mean_mass("Xe2Pb", _DEFAULT_PRESET) > 0.0


def test_gui_axis_label_follows_charge_metadata():
    assert _axis_label_for_result(None) == "mass"
    assert _axis_label_for_result({"metadata": {"axis_unit": "m/z"}}) == "m/z"


def test_gui_resolving_power_sigma_conversion_round_trips():
    mean_mass = _single_formula_mean_mass(
        "C500H800N125O200S10",
        "common",
    )

    sigma = _sigma_from_resolving_power(mean_mass, 100_000)
    resolving_power = _resolving_power_from_sigma(mean_mass, sigma)

    assert sigma == pytest.approx(0.0513123459)
    assert resolving_power == pytest.approx(100_000)


def test_gui_mean_mass_includes_mass_only_elements():
    base = _single_formula_mean_mass("C6H12O6", "common")
    iodinated = _single_formula_mean_mass("C6H12O6I", "common")

    assert iodinated - base == pytest.approx(126.9044719)


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
