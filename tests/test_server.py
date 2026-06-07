import pytest
import numpy as np

from fastiso.server import _TABLE_CACHE, simulate_window
from fastiso import has_cython_backend, load_isotope_registry


def test_simulate_window_returns_json_serializable_profile():
    _TABLE_CACHE.clear()

    response = simulate_window({
        "formula": "C6H12O6",
        "preset": "common",
        "table_dm": 0.01,
        "min_fft_len": 2048,
        "resolving_power": 100_000,
        "window": {"mode": "residual", "start": -0.1, "stop": 0.1},
        "output_dm": 0.01,
        "method": "cython_log_pruned",
    })

    assert response["formula"] == "C6H12O6"
    assert len(response["mass_axis"]) == len(response["intensity"])
    assert response["metadata"]["n_points"] == len(response["intensity"])
    assert response["metadata"]["isotope_data_version"] == "common-isospecpy-prototype-v1"
    assert response["metadata"]["cache_key"]["elements"][0] == "H"
    assert response["metadata"]["active_fraction"] > 0.0
    assert response["summary"]["apex_intensity"] > 0.0
    assert len(_TABLE_CACHE) == 1


def test_simulate_window_supports_absolute_mass_window():
    _TABLE_CACHE.clear()

    response = simulate_window({
        "formula": "C6H12O6",
        "elements": ["C", "H", "O"],
        "table_dm": 0.01,
        "min_fft_len": 2048,
        "resolving_power": 100_000,
        "window": {"mode": "mass", "start": 180.0, "stop": 180.2},
        "output_dm": 0.01,
        "method": "log_pruned",
    })

    assert response["elements"] == ["C", "H", "O"]
    assert response["mass_axis"][0] == 180.0
    assert response["mass_axis"][-1] <= 180.205
    assert response["metadata"]["cache_key"]["elements"] == ["C", "H", "O"]


def test_simulate_window_default_auto_uses_production_storage_when_available():
    _TABLE_CACHE.clear()

    response = simulate_window({
        "formula": "C6H12O6",
        "elements": ["C", "H", "O"],
        "table_dm": 0.01,
        "min_fft_len": 2048,
        "resolving_power": 100_000,
        "window": {"mode": "residual", "start": -0.05, "stop": 0.05},
        "output_dm": 0.01,
        "workers": 2,
    })

    expected_storage = "production" if has_cython_backend() else "research"
    expected_dtype = "float32" if has_cython_backend() else "float64"
    assert response["metadata"]["method"] == "cython_auto"
    assert isinstance(response["metadata"]["selected_method"], str)
    assert response["metadata"]["workers"] == 2
    assert response["metadata"]["table_storage"] == expected_storage
    assert response["metadata"]["attenuation_dtype"] == expected_dtype
    assert response["metadata"]["table_nbytes"] > 0


def test_simulate_window_treats_monoisotopic_elements_as_mass_shift():
    _TABLE_CACHE.clear()
    registry = load_isotope_registry()
    iodine_mass = registry.patterns["I"].monoisotopic_mass
    payload = {
        "elements": ["C", "H", "O"],
        "table_dm": 0.01,
        "min_fft_len": 2048,
        "gaussian_sigma": 0.001,
        "window": {"mode": "residual", "start": -0.1, "stop": 0.1},
        "output_dm": 0.01,
        "method": "log_pruned",
    }

    base = simulate_window({"formula": "C6H12O6", **payload})
    iodinated = simulate_window({"formula": "C6H12O6I", **payload})

    assert iodinated["elements"] == ["C", "H", "O"]
    assert iodinated["metadata"]["cache_key"]["elements"] == ["C", "H", "O"]
    assert iodinated["metadata"]["mass_only_counts"] == {"I": 1}
    assert iodinated["metadata"]["mass_shift"] == pytest.approx(iodine_mass)
    np.testing.assert_allclose(iodinated["intensity"], base["intensity"])
    np.testing.assert_allclose(
        np.asarray(iodinated["mass_axis"]) - np.asarray(base["mass_axis"]),
        iodine_mass,
    )
    assert len(_TABLE_CACHE) == 1


def test_simulate_window_rejects_unselected_multiisotope_formula_element():
    with pytest.raises(ValueError, match="not available"):
        simulate_window({
            "formula": "C6H12O6Cl",
            "elements": ["C", "H", "O"],
            "table_dm": 0.01,
            "min_fft_len": 2048,
            "window": {"mode": "residual", "start": -0.1, "stop": 0.1},
            "output_dm": 0.01,
            "method": "log_pruned",
        })


def test_simulate_window_supports_all_mass_only_formula():
    response = simulate_window({
        "formula": "NaI",
        "preset": "common",
        "table_dm": 0.01,
        "min_fft_len": 2048,
        "gaussian_sigma": 0.002,
        "window": {"mode": "mass", "start": 149.8, "stop": 150.1},
        "output_dm": 0.01,
        "method": "log_pruned",
    })

    assert response["elements"] == []
    assert response["metadata"]["cache_key"]["elements"] == []
    assert response["metadata"]["mass_only_counts"] == {"Na": 1, "I": 1}
    assert response["summary"]["apex_intensity"] > 0.0


def test_simulate_window_full_preset_uses_full_isotope_resource():
    _TABLE_CACHE.clear()

    response = simulate_window({
        "formula": "Xe2Pb",
        "preset": "full",
        "table_dm": 0.01,
        "min_fft_len": 2048,
        "gaussian_sigma": 0.002,
        "window": {"mode": "residual", "start": -0.1, "stop": 0.1},
        "output_dm": 0.01,
        "method": "log_pruned",
    })

    assert response["preset"] == "full"
    assert response["metadata"]["isotope_resource"] == "full"
    assert response["metadata"]["isotope_data_version"] == (
        "full-natural-abundance-isospecpy-prototype-v1"
    )
    assert response["elements"] == ["Xe", "Pb"]
    assert response["summary"]["apex_intensity"] > 0.0


def test_simulate_window_full_preset_rejects_no_stable_element():
    with pytest.raises(ValueError, match="missing"):
        simulate_window({
            "formula": "Bi",
            "preset": "full",
            "table_dm": 0.01,
            "min_fft_len": 2048,
            "window": {"mode": "residual", "start": -0.1, "stop": 0.1},
            "output_dm": 0.01,
            "method": "log_pruned",
        })


def test_fastapi_simulate_window_endpoint_when_available():
    try:
        from fastapi.testclient import TestClient
    except (ImportError, RuntimeError) as exc:
        pytest.skip(f"FastAPI TestClient is unavailable: {exc}")
    from fastiso.server import app

    assert app is not None
    client = TestClient(app)
    response = client.post(
        "/simulate/window",
        json={
            "formula": "C6H12O6",
            "elements": ["C", "H", "O"],
            "table_dm": 0.01,
            "min_fft_len": 2048,
            "resolving_power": 100_000,
            "window": {"mode": "residual", "start": -0.05, "stop": 0.05},
            "output_dm": 0.01,
            "method": "log_pruned",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["metadata"]["transform"] == "czt_irfft_window"
    assert len(payload["mass_axis"]) == len(payload["intensity"])
