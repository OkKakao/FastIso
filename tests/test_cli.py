import csv
import json
import math

import pytest

from fastiso.cli import main


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
