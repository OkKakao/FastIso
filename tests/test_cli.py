import csv
import json
import math

import numpy as np
import pytest

from fastiso.cli import main, simulate_profiles


def test_cli_lists_full_isotope_resource(capsys):
    assert main(["isotopes", "list", "--preset", "full", "--format", "json"]) == 0

    payload = json.loads(capsys.readouterr().out)

    assert payload["preset"] == "full"
    assert payload["resource"] == "full"
    assert len(payload["elements"]) == 80
    assert {row["element"] for row in payload["elements"]}.issuperset(
        {"C", "H", "O", "K", "Xe"}
    )


def test_cli_inspects_long_lived_natural_isotope(capsys):
    assert main(["isotopes", "inspect", "K", "--preset", "full", "--format", "json"]) == 0

    payload = json.loads(capsys.readouterr().out)

    assert payload["element"] == "K"
    assert payload["resource"] == "full"
    assert len(payload["isotopes"]) == 3


def test_cli_window_csv_outputs_regular_profile_rows(capsys):
    assert main(
        [
            "window",
            "C6H12O6",
            "--elements",
            "C",
            "H",
            "O",
            "--dm",
            "0.01",
            "--min-fft-len",
            "255",
            "--start",
            "-0.02",
            "--stop",
            "0.02",
            "--output-dm",
            "0.02",
            "--method",
            "log_pruned",
            "--storage-mode",
            "research",
        ]
    ) == 0

    rows = list(csv.DictReader(capsys.readouterr().out.splitlines()))

    assert len(rows) == 3
    assert {row["formula"] for row in rows} == {"C6H12O6"}
    assert all(math.isfinite(float(row["intensity"])) for row in rows)
    assert [round(float(row["mass"]), 2) for row in rows] == sorted(
        round(float(row["mass"]), 2) for row in rows
    )


def test_cli_window_json_keeps_mass_only_elements_as_shifts(capsys):
    assert main(
        [
            "window",
            "C6H12O6I",
            "--elements",
            "C",
            "H",
            "O",
            "--dm",
            "0.01",
            "--min-fft-len",
            "255",
            "--start",
            "306.90",
            "--stop",
            "306.94",
            "--window-mode",
            "mass",
            "--output-dm",
            "0.02",
            "--method",
            "log_pruned",
            "--storage-mode",
            "research",
            "--format",
            "json",
        ]
    ) == 0

    payload = json.loads(capsys.readouterr().out)
    metadata = payload["metadata"]

    assert metadata["spectral_elements"] == ["C", "H", "O"]
    assert metadata["mass_shifts"][0] > 126.0
    assert metadata["window_mode"] == "mass"
    assert payload["mass_axis"][0] == pytest.approx([306.90, 306.92, 306.94])


def test_cli_accepts_bracketed_formula_expression(capsys):
    assert main(
        [
            "window",
            "(CH3OH)2(HCl)2",
            "--elements",
            "C",
            "H",
            "O",
            "Cl",
            "--dm",
            "0.01",
            "--min-fft-len",
            "255",
            "--start",
            "-0.01",
            "--stop",
            "0.01",
            "--output-dm",
            "0.01",
            "--method",
            "log_pruned",
            "--storage-mode",
            "research",
            "--format",
            "json",
        ]
    ) == 0

    payload = json.loads(capsys.readouterr().out)

    assert payload["formulas"] == ["(CH3OH)2(HCl)2"]
    assert payload["metadata"]["spectral_elements"] == ["C", "H", "O", "Cl"]


def test_cli_adaptive_window_includes_skewed_small_formula_base_peak(capsys):
    assert main(
        [
            "window",
            "S10",
            "--elements",
            "S",
            "--window-mode",
            "adaptive",
            "--dm",
            "0.05",
            "--min-fft-len",
            "255",
            "--method",
            "log_pruned",
            "--storage-mode",
            "research",
            "--format",
            "json",
        ]
    ) == 0

    payload = json.loads(capsys.readouterr().out)
    metadata = payload["metadata"]
    mean_mass = metadata["total_mean_masses"][0]
    residual_axis = [mass - mean_mass for mass in payload["mass_axis"][0]]
    max_idx = max(
        range(len(payload["intensity"][0])),
        key=lambda idx: payload["intensity"][0][idx],
    )

    assert metadata["window_mode"] == "adaptive"
    assert metadata["window_start"] < -0.928
    assert metadata["window_stop"] > 1.068
    assert residual_axis[max_idx] == pytest.approx(-0.928, abs=0.05)
    assert metadata["output_dm"] == pytest.approx(0.05)


def test_adaptive_window_uses_exact_support_for_chlorine_series():
    formulas = ["Cl"] + [f"Cl{count}" for count in range(2, 7)]

    result = simulate_profiles(
        formulas,
        elements=["Cl"],
        window_mode="adaptive",
        dm=0.1,
        output_dm=0.1,
        min_fft_len=255,
        method="log_pruned",
        storage_mode="research",
    )
    metadata = result["metadata"]

    assert metadata["auto_window_method"] == "exact_support"
    assert metadata["profile_backend"] == "exact_gaussian"
    assert metadata["window_start"] == pytest.approx(-3.004571, abs=1e-4)
    assert metadata["window_stop"] == pytest.approx(9.177728, abs=1e-4)

    mean_masses = metadata["total_mean_masses"]
    for row_idx, formula in enumerate(formulas):
        intensity = result["intensity"][row_idx]
        count = row_idx + 1
        residual_axis = [
            mass - mean_masses[row_idx]
            for mass in result["mass_axis"][row_idx]
        ]
        light_peak = count * -0.4840951867793653
        heavy_peak = count * 1.5129547232206347
        assert min(residual_axis) <= light_peak
        assert max(residual_axis) >= heavy_peak
        assert np.min(intensity) >= 0.0
        assert np.sum(intensity) == pytest.approx(1.0)


def test_adaptive_window_skips_exact_backend_for_large_default_formula():
    result = simulate_profiles(
        ["C500H800N125O200S10"],
        elements=["C", "H", "N", "O", "S"],
        window_mode="auto",
        auto_grid=True,
        min_fft_len="auto",
        method="log_pruned",
        storage_mode="research",
    )
    metadata = result["metadata"]

    assert metadata["requested_window_mode"] == "auto"
    assert metadata["window_mode"] == "adaptive"
    assert metadata["auto_window_method"] == "sigma"
    assert metadata["profile_backend"] == "ft"
    assert metadata["auto_min_fft_len"] is True
    assert metadata["min_fft_len"] == 255
    assert metadata["n_fft"] < 32768
    assert "exact_state_counts" not in metadata


def test_cli_auto_grid_reduces_single_atom_ringing(capsys):
    assert main(
        [
            "window",
            "S",
            "--elements",
            "S",
            "--start",
            "-0.100",
            "--stop",
            "-0.085",
            "--auto-grid",
            "--samples-per-fwhm",
            "8",
            "--min-fft-len",
            "32768",
            "--method",
            "log_pruned",
            "--storage-mode",
            "research",
            "--format",
            "json",
        ]
    ) == 0

    payload = json.loads(capsys.readouterr().out)
    intensity = payload["intensity"][0]
    metadata = payload["metadata"]

    assert metadata["auto_grid"] is True
    assert metadata["dm"] < 0.0001
    assert metadata["output_dm"] == pytest.approx(metadata["dm"])
    assert metadata["profile_backend"] == "exact_gaussian"
    assert min(intensity) >= 0.0
    assert max(intensity) > 0.1


def test_cli_gaussian_sigma_disables_default_resolving_power(capsys):
    assert main(
        [
            "window",
            "C6H12O6",
            "--elements",
            "C",
            "H",
            "O",
            "--dm",
            "0.01",
            "--min-fft-len",
            "255",
            "--start",
            "-0.01",
            "--stop",
            "0.01",
            "--output-dm",
            "0.01",
            "--gaussian-sigma",
            "0.005",
            "--method",
            "log_pruned",
            "--storage-mode",
            "research",
            "--format",
            "json",
        ]
    ) == 0

    metadata = json.loads(capsys.readouterr().out)["metadata"]

    assert metadata["resolving_power"] is None
    assert metadata["gaussian_sigma"] == pytest.approx(0.005)


def test_cli_simulate_can_write_csv_file(tmp_path):
    output_path = tmp_path / "profile.csv"

    assert main(
        [
            "simulate",
            "C2H4O",
            "--elements",
            "C",
            "H",
            "O",
            "--dm",
            "0.02",
            "--min-fft-len",
            "255",
            "--method",
            "log_pruned",
            "--storage-mode",
            "research",
            "--output",
            str(output_path),
        ]
    ) == 0

    rows = list(csv.DictReader(output_path.read_text(encoding="utf-8").splitlines()))

    assert len(rows) >= 255
    assert rows[0]["formula"] == "C2H4O"
