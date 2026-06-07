"""Versioned isotope-data loading and element presets."""

from __future__ import annotations

from dataclasses import dataclass
import json
from importlib import resources
from typing import Mapping, Sequence

import numpy as np

from .formula import parse_formula


ELEMENT_PRESETS: Mapping[str, tuple[str, ...]] = {
    "bio": ("C", "H", "N", "O", "P", "S", "Se"),
    "organic": ("C", "H", "N", "O", "P", "S", "F", "Cl", "Br", "I", "B", "Si"),
    "halogen": ("F", "Cl", "Br", "I"),
    "adduct": ("H", "Li", "Na", "K", "Mg", "Al", "Ca"),
    "common": (
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
    ),
}


@dataclass(frozen=True)
class IsotopePattern:
    element: str
    masses: np.ndarray
    abundances: np.ndarray

    @property
    def mean_mass(self) -> float:
        return float(np.dot(self.masses, self.abundances))

    @property
    def variance(self) -> float:
        centered = self.masses - self.mean_mass
        return float(np.dot(centered * centered, self.abundances))

    @property
    def is_mass_only(self) -> bool:
        """Return whether this pattern contributes only a deterministic mass."""

        return self.masses.size == 1 or self.variance <= 1e-24

    @property
    def monoisotopic_mass(self) -> float:
        """Return the deterministic mass for mass-only isotope patterns."""

        if not self.is_mass_only:
            raise ValueError(f"{self.element} has multiple natural isotopes")
        return self.mean_mass


@dataclass(frozen=True)
class FormulaIsotopeComponents:
    """Formula split into spectral and deterministic-mass isotope components."""

    counts: Mapping[str, int]
    spectral_elements: tuple[str, ...]
    spectral_counts: Mapping[str, int]
    mass_only_counts: Mapping[str, int]
    mass_shift: float


@dataclass(frozen=True)
class IsotopeRegistry:
    """Loaded isotope dataset plus named element presets."""

    source: str
    version: str
    patterns: Mapping[str, IsotopePattern]
    presets: Mapping[str, tuple[str, ...]]

    @classmethod
    def load(cls, resource: str = "common") -> "IsotopeRegistry":
        """Load a packaged isotope dataset by resource name."""

        if not resource.endswith(".json"):
            resource = f"{resource}.json"
        path = resources.files("fastiso.isotope_data").joinpath(resource)
        raw = json.loads(path.read_text(encoding="utf-8"))
        patterns = _patterns_from_raw(raw)
        presets = dict(ELEMENT_PRESETS)
        presets["full"] = tuple(patterns)
        return cls(
            source=str(raw["source"]),
            version=str(raw["version"]),
            patterns=patterns,
            presets=presets,
        )

    def elements_for_preset(self, preset: str) -> tuple[str, ...]:
        """Return element symbols for a named preset."""

        try:
            return self.presets[preset]
        except KeyError as exc:
            names = ", ".join(sorted(self.presets))
            raise ValueError(f"unknown element preset {preset!r}; choose one of {names}") from exc

    def isotope_patterns(
        self,
        *,
        preset: str = "common",
        elements: Sequence[str] | None = None,
    ) -> dict[str, IsotopePattern]:
        """Return isotope patterns for explicit elements or a preset."""

        selected = tuple(elements) if elements is not None else self.elements_for_preset(preset)
        missing = [element for element in selected if element not in self.patterns]
        if missing:
            raise ValueError(
                "isotope data are missing for element(s): " + ", ".join(missing)
            )
        return {element: self.patterns[element] for element in selected}


def load_isotope_registry(resource: str = "common") -> IsotopeRegistry:
    """Load a packaged isotope registry."""

    return IsotopeRegistry.load(resource)


def load_isotope_patterns(
    *,
    preset: str = "common",
    elements: Sequence[str] | None = None,
    resource: str | None = None,
) -> dict[str, IsotopePattern]:
    """Load normalized isotope patterns for a preset or explicit element list."""

    registry = load_isotope_registry(_default_resource_for_request(preset, resource))
    return registry.isotope_patterns(preset=preset, elements=elements)


def default_isotope_patterns() -> dict[str, IsotopePattern]:
    """Return normalized natural isotope patterns for the common preset."""

    return load_isotope_patterns(preset="common")


def isotope_data_version(resource: str = "common") -> str:
    """Return the version string for a packaged isotope dataset."""

    return load_isotope_registry(resource).version


def _default_resource_for_request(preset: str, resource: str | None) -> str:
    if resource is not None:
        return resource
    if preset == "full":
        return "full"
    return "common"


def split_formula_isotope_components(
    formula: str | Mapping[str, int],
    patterns: Mapping[str, IsotopePattern],
    *,
    elements: Sequence[str] | None = None,
    allow_unselected_mass_only: bool = True,
) -> FormulaIsotopeComponents:
    """Split a formula into FT-spectral counts and mass-only shifts.

    Single-isotope elements such as F, Na, P, and I have zero isotope variance
    in the packaged natural-abundance data. They do not need log/phase table
    rows; they only shift the final mass axis.
    """

    counts = _validated_formula_counts(formula)
    selected = tuple(elements) if elements is not None else tuple(patterns)
    missing_selected = [element for element in selected if element not in patterns]
    if missing_selected:
        raise ValueError(
            "isotope data are missing for element(s): " + ", ".join(missing_selected)
        )

    missing_data = [element for element in counts if element not in patterns]
    if missing_data:
        raise ValueError(
            "isotope data are missing for element(s): " + ", ".join(missing_data)
        )

    selected_set = set(selected)
    unavailable: list[str] = []
    for element, count in counts.items():
        if count == 0:
            continue
        if element in selected_set:
            continue
        if allow_unselected_mass_only and patterns[element].is_mass_only:
            continue
        unavailable.append(element)
    if unavailable:
        raise ValueError(
            "element(s) not available in selected isotope elements: "
            + ", ".join(unavailable)
        )

    spectral_elements = tuple(
        element
        for element in selected
        if counts.get(element, 0) != 0 and not patterns[element].is_mass_only
    )
    spectral_counts = {element: int(counts[element]) for element in spectral_elements}
    mass_only_counts = {
        element: int(count)
        for element, count in counts.items()
        if count != 0 and patterns[element].is_mass_only
    }
    mass_shift = sum(
        count * patterns[element].monoisotopic_mass
        for element, count in mass_only_counts.items()
    )
    return FormulaIsotopeComponents(
        counts=counts,
        spectral_elements=spectral_elements,
        spectral_counts=spectral_counts,
        mass_only_counts=mass_only_counts,
        mass_shift=float(mass_shift),
    )


def _patterns_from_raw(raw: Mapping[str, object]) -> dict[str, IsotopePattern]:
    elements = raw.get("elements")
    if not isinstance(elements, dict):
        raise ValueError("isotope dataset must contain an elements object")
    patterns: dict[str, IsotopePattern] = {}
    for element, rows in elements.items():
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"isotope data for {element!r} must be a non-empty list")
        masses = np.array([float(row["mass"]) for row in rows], dtype=np.float64)
        abundances = np.array([float(row["abundance"]) for row in rows], dtype=np.float64)
        abundance_sum = abundances.sum()
        if abundance_sum <= 0.0:
            raise ValueError(f"invalid isotope abundances for {element!r}")
        patterns[str(element)] = IsotopePattern(
            element=str(element),
            masses=masses,
            abundances=abundances / abundance_sum,
        )
    return patterns


def _validated_formula_counts(formula: str | Mapping[str, int]) -> dict[str, int]:
    counts = parse_formula(formula) if isinstance(formula, str) else dict(formula)
    for element, count in counts.items():
        if int(count) != count:
            raise ValueError(f"non-integer count for element {element!r}: {count}")
        if count < 0:
            raise ValueError(f"negative count for element {element!r}: {count}")
        counts[element] = int(count)
    return counts
