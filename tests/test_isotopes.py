import json
from importlib import resources

import numpy as np
import pytest

from fastiso import (
    CenteredLogPhaseTable,
    isotope_data_version,
    load_isotope_patterns,
    load_isotope_registry,
    split_formula_isotope_components,
)


COMMON_ELEMENTS = (
    "H",
    "Li",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "K",
    "Ca",
    "Fe",
    "Ni",
    "Cu",
    "Zn",
    "Se",
    "Br",
    "I",
)
FULL_EXCLUDED_ELEMENTS = {"E", "Me", "Pn", "Bi", "Th", "Pa", "U", "Nh"}


def test_common_registry_loads_versioned_dataset():
    registry = load_isotope_registry()

    assert registry.version == "common-isospecpy-prototype-v1"
    assert "IsoSpecPy.PeriodicTbl" in registry.source
    assert {"bio", "organic", "halogen", "adduct", "common", "full"}.issubset(
        registry.presets
    )
    assert registry.presets["common"] == COMMON_ELEMENTS
    assert "Al" in registry.presets["adduct"]
    assert set(COMMON_ELEMENTS).issubset(registry.patterns)


def test_load_isotope_patterns_supports_presets_and_explicit_elements():
    bio = load_isotope_patterns(preset="bio")
    halogen = load_isotope_patterns(preset="halogen")
    explicit = load_isotope_patterns(elements=("C", "Na", "Ni"))
    full = load_isotope_patterns(preset="full")

    assert {"C", "H", "N", "O", "P", "S"}.issubset(bio)
    assert set(halogen) == {"F", "Cl", "Br", "I"}
    assert set(explicit) == {"C", "Na", "Ni"}
    assert len(full) == 80
    assert {"H", "He", "Xe", "Pb"}.issubset(full)
    assert FULL_EXCLUDED_ELEMENTS.isdisjoint(full)


def test_full_registry_loads_stable_element_dataset():
    registry = load_isotope_registry(resource="full")

    assert registry.version == "full-strict-stable-isotopes-isospecpy-iaea-prototype-v1"
    assert "half_life == STABLE" in registry.source
    assert len(registry.patterns) == 80
    assert registry.presets["full"] == tuple(registry.patterns)
    assert {"H", "He", "Be", "Xe", "Pb"}.issubset(registry.patterns)
    assert FULL_EXCLUDED_ELEMENTS.isdisjoint(registry.patterns)


def test_common_isotope_patterns_are_normalized_and_ordered():
    registry = load_isotope_registry()

    for element in COMMON_ELEMENTS:
        pattern = registry.patterns[element]

        assert pattern.element == element
        assert pattern.masses.ndim == 1
        assert pattern.abundances.shape == pattern.masses.shape
        assert np.all(np.isfinite(pattern.masses))
        assert np.all(pattern.abundances >= 0.0)
        assert pattern.abundances.sum() == pytest.approx(1.0)
        assert np.all(np.diff(pattern.masses) > 0.0)
        assert pattern.mean_mass > 0.0
        assert pattern.variance >= 0.0


def test_common_isotope_patterns_match_isospecpy_source_when_available():
    isospecpy = pytest.importorskip("IsoSpecPy")
    registry = load_isotope_registry()
    table = isospecpy.PeriodicTbl

    for element in COMMON_ELEMENTS:
        pattern = registry.patterns[element]
        expected_masses = np.asarray(table.symbol_to_masses[element], dtype=np.float64)
        expected_probs = np.asarray(table.symbol_to_probs[element], dtype=np.float64)
        expected_probs = expected_probs / expected_probs.sum()

        np.testing.assert_allclose(pattern.masses, expected_masses, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(pattern.abundances, expected_probs, rtol=1e-15, atol=1e-15)


def test_full_isotope_patterns_match_isospecpy_source_when_available():
    isospecpy = pytest.importorskip("IsoSpecPy")
    registry = load_isotope_registry(resource="full")
    table = isospecpy.PeriodicTbl
    removed = _full_removed_radioactive_isotopes()

    for element, pattern in registry.patterns.items():
        mass_numbers = np.asarray(table.symbol_to_massNo[element], dtype=np.int64)
        keep = ~np.isin(mass_numbers, removed.get(element, ()))
        expected_masses = np.asarray(table.symbol_to_masses[element], dtype=np.float64)[keep]
        expected_probs = np.asarray(table.symbol_to_probs[element], dtype=np.float64)[keep]
        expected_probs = expected_probs / expected_probs.sum()

        np.testing.assert_allclose(pattern.masses, expected_masses, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(pattern.abundances, expected_probs, rtol=1e-15, atol=1e-15)


def test_full_isotope_patterns_remove_radioactive_natural_rows():
    registry = load_isotope_registry(resource="full")
    removed = _full_removed_radioactive_isotopes()

    assert len(removed) == 28
    assert sum(len(isotopes) for isotopes in removed.values()) == 38
    assert len(registry.patterns) == 80
    assert sum(pattern.masses.size for pattern in registry.patterns.values()) == 244
    assert removed["K"] == [40]
    assert removed["Ca"] == [48]
    assert removed["Xe"] == [124, 134, 136]
    assert 40 not in _isotope_numbers(registry, "K")
    assert 48 not in _isotope_numbers(registry, "Ca")
    assert 136 not in _isotope_numbers(registry, "Xe")
    assert 208 in _isotope_numbers(registry, "Pb")


@pytest.mark.parametrize("element", COMMON_ELEMENTS)
def test_single_element_formula_splits_into_spectral_or_mass_only(element):
    registry = load_isotope_registry()
    pattern = registry.patterns[element]

    components = split_formula_isotope_components(
        f"{element}3",
        registry.patterns,
        elements=COMMON_ELEMENTS,
    )

    if pattern.is_mass_only:
        assert components.spectral_elements == ()
        assert components.spectral_counts == {}
        assert components.mass_only_counts == {element: 3}
        assert components.mass_shift == pytest.approx(3 * pattern.monoisotopic_mass)
    else:
        assert components.spectral_elements == (element,)
        assert components.spectral_counts == {element: 3}
        assert components.mass_only_counts == {}
        assert components.mass_shift == 0.0


@pytest.mark.parametrize("element", COMMON_ELEMENTS)
def test_single_multiisotope_element_table_matches_direct_rebuild(element):
    registry = load_isotope_registry()
    if registry.patterns[element].is_mass_only:
        pytest.skip(f"{element} is deterministic mass-only")

    table = CenteredLogPhaseTable.build(
        elements=(element,),
        dm=0.01,
        min_fft_len=2048,
    )
    counts = table.counts_from_formulas([f"{element}5"])

    log_spectrum = table.residual_spectrum_many_counts(counts)
    direct_spectrum = table.residual_spectrum_many_counts(
        counts,
        method="direct_rebuild",
    )
    mass_axis, profiles, info = table.mass_profile_many_counts(counts)

    assert info["n_fft"] == table.n_fft
    assert mass_axis.shape == profiles.shape
    np.testing.assert_allclose(profiles.sum(axis=-1), np.array([1.0]), atol=1e-12)
    assert _relative_l2(log_spectrum, direct_spectrum) < 1e-12


def test_unknown_preset_is_rejected():
    registry = load_isotope_registry()

    with pytest.raises(ValueError, match="unknown element preset"):
        registry.isotope_patterns(preset="not-a-preset")


def test_split_formula_isotope_components_moves_monoisotopic_elements_to_mass_shift():
    registry = load_isotope_registry()

    components = split_formula_isotope_components(
        "C6H12O6I",
        registry.patterns,
        elements=("C", "H", "O"),
    )

    assert registry.patterns["I"].is_mass_only
    assert components.spectral_elements == ("C", "H", "O")
    assert components.spectral_counts == {"C": 6, "H": 12, "O": 6}
    assert components.mass_only_counts == {"I": 1}
    assert components.mass_shift == pytest.approx(
        registry.patterns["I"].monoisotopic_mass
    )


def test_split_formula_isotope_components_handles_aluminum_and_nickel():
    registry = load_isotope_registry()

    components = split_formula_isotope_components(
        "C2H4AlNi",
        registry.patterns,
        elements=("C", "H", "Ni"),
    )

    assert registry.patterns["Al"].is_mass_only
    assert not registry.patterns["Ni"].is_mass_only
    assert components.spectral_elements == ("C", "H", "Ni")
    assert components.spectral_counts == {"C": 2, "H": 4, "Ni": 1}
    assert components.mass_only_counts == {"Al": 1}
    assert components.mass_shift == pytest.approx(
        registry.patterns["Al"].monoisotopic_mass
    )


def test_split_formula_isotope_components_rejects_unselected_multiisotope_elements():
    registry = load_isotope_registry()

    with pytest.raises(ValueError, match="not available"):
        split_formula_isotope_components(
            "C6H12O6Cl",
            registry.patterns,
            elements=("C", "H", "O"),
        )


def test_table_cache_key_includes_isotope_data_version():
    table = CenteredLogPhaseTable.build(
        elements=("C", "H", "O"),
        dm=0.01,
        min_fft_len=2048,
    )

    assert table.cache_key.elements == ("C", "H", "O")
    assert table.cache_key.dm == 0.01
    assert table.cache_key.n_fft == table.n_fft
    assert table.cache_key.isotope_data_version == isotope_data_version()


def _relative_l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def _full_removed_radioactive_isotopes() -> dict[str, list[int]]:
    raw = _full_raw_json()
    selection = raw["selection"]
    return {
        element: [int(isotope) for isotope in isotopes]
        for element, isotopes in selection["removed_radioactive_isotopes"].items()
    }


def _isotope_numbers(registry, element: str) -> set[int]:
    raw = _full_raw_json()
    return {int(row["isotope"]) for row in raw["elements"][element]}


def _full_raw_json() -> dict:
    path = resources.files("fastiso.isotope_data").joinpath("full.json")
    return json.loads(path.read_text(encoding="utf-8"))
