"""Centered log/phase isotope characteristic tables."""

from __future__ import annotations

from dataclasses import dataclass
from math import log, pi, sqrt
from typing import Mapping, Sequence

import numpy as np

from .cache import TableCacheKey, make_table_cache_key
from .fast_odd_irfft import (
    czt_irfft_window,
    fast_odd_irfft,
    positive_frequency_count,
    rfftfreq_odd,
    suggest_fast_odd_len,
)
from .formula import counts_matrix
from .isotopes import (
    IsotopePattern,
    default_isotope_patterns,
    isotope_data_version as default_isotope_data_version,
)

try:
    from . import _cython_log_table
except ImportError:
    _cython_log_table = None


def has_cython_backend() -> bool:
    """Return whether the optional compiled log-table backend is available."""

    return _cython_log_table is not None


@dataclass(frozen=True)
class CenteredLogPhaseTable:
    """Precomputed positive-frequency centered log/phase resource."""

    elements: tuple[str, ...]
    isotope_patterns: tuple[IsotopePattern, ...]
    dm: float
    n_fft: int
    omega: np.ndarray
    attenuation: np.ndarray
    phase: np.ndarray
    phase_cycles: np.ndarray
    phase_uint64: np.ndarray
    attenuation_count_threshold: np.ndarray
    attenuation_template_trigger_count: np.ndarray
    mean_masses: np.ndarray
    variances: np.ndarray
    template_floor: float = 1e-15
    isotope_data_version: str = "unknown"
    threshold_prune_cutoff: float = 1e-12
    storage_mode: str = "research"

    @classmethod
    def build(
        cls,
        *,
        elements: Sequence[str] = ("C", "H", "N", "O", "S", "Cl", "Br"),
        dm: float = 0.001,
        n_fft: int | None = None,
        min_fft_len: int = 32768,
        template_floor: float = 1e-15,
        attenuation_dtype: str | np.dtype | None = None,
        storage_mode: str = "research",
        isotope_patterns: Mapping[str, IsotopePattern] | None = None,
        isotope_data_version: str | None = None,
    ) -> "CenteredLogPhaseTable":
        """Build a centered log/phase table for a fixed mass grid."""

        if dm <= 0.0:
            raise ValueError("dm must be positive")
        if not (0.0 < template_floor < 1.0):
            raise ValueError("template_floor must be between 0 and 1")

        n_fft = suggest_fast_odd_len(min_fft_len) if n_fft is None else int(n_fft)
        if n_fft < 1 or n_fft % 2 == 0:
            raise ValueError(f"n_fft must be positive and odd, got {n_fft}")
        if storage_mode not in {"research", "production", "minimal"}:
            raise ValueError("storage_mode must be 'research', 'production', or 'minimal'")
        if attenuation_dtype is None:
            attenuation_dtype = np.float32 if storage_mode != "research" else np.float64
        attenuation_dtype = np.dtype(attenuation_dtype)
        if attenuation_dtype not in {np.dtype(np.float32), np.dtype(np.float64)}:
            raise ValueError("attenuation_dtype must be float32 or float64")

        if isotope_patterns is None:
            patterns_by_element = default_isotope_patterns()
            data_version = isotope_data_version or default_isotope_data_version()
        else:
            patterns_by_element = isotope_patterns
            data_version = isotope_data_version or "custom"
        element_tuple = tuple(elements)
        patterns = tuple(patterns_by_element[element] for element in element_tuple)
        frequencies = rfftfreq_odd(n_fft, d=dm)
        omega = 2.0 * pi * frequencies

        attenuation_rows: list[np.ndarray] = []
        phase_rows: list[np.ndarray] = []
        mean_masses: list[float] = []
        variances: list[float] = []
        for pattern in patterns:
            attenuation, phase, mean_mass = _build_element_log_phase(
                pattern,
                omega,
                template_floor=template_floor,
            )
            attenuation_rows.append(attenuation)
            phase_rows.append(phase)
            mean_masses.append(mean_mass)
            variances.append(pattern.variance)

        if attenuation_rows:
            attenuation = np.stack(attenuation_rows).astype(attenuation_dtype, copy=False)
            phase = np.stack(phase_rows)
        else:
            attenuation = np.zeros((0, omega.shape[0]), dtype=attenuation_dtype)
            phase = np.zeros((0, omega.shape[0]), dtype=np.float64)
        phase_cycles = _wrap_phase_cycles(phase)
        phase_uint64 = _phase_cycles_to_uint64(phase_cycles)
        threshold_prune_cutoff = 1e-12
        attenuation_count_threshold = _attenuation_count_threshold(
            attenuation,
            prune_cutoff=threshold_prune_cutoff,
        )
        attenuation_template_trigger_count = _attenuation_template_trigger_count(
            attenuation_count_threshold,
            active_fraction_threshold=0.25,
        )
        if storage_mode in {"production", "minimal"}:
            phase_cycles = np.empty((0, 0), dtype=np.float64)
        if storage_mode == "minimal":
            phase = np.empty((0, 0), dtype=np.float64)

        return cls(
            elements=element_tuple,
            isotope_patterns=patterns,
            dm=float(dm),
            n_fft=n_fft,
            omega=omega,
            attenuation=attenuation,
            phase=phase,
            phase_cycles=phase_cycles,
            phase_uint64=phase_uint64,
            attenuation_count_threshold=attenuation_count_threshold,
            attenuation_template_trigger_count=attenuation_template_trigger_count,
            mean_masses=np.array(mean_masses, dtype=np.float64),
            variances=np.array(variances, dtype=np.float64),
            template_floor=float(template_floor),
            isotope_data_version=data_version,
            threshold_prune_cutoff=threshold_prune_cutoff,
            storage_mode=storage_mode,
        )

    @classmethod
    def build_for_counts(
        cls,
        counts: np.ndarray,
        *,
        elements: Sequence[str] = ("C", "H", "N", "O", "S", "Cl", "Br"),
        dm: float = 0.001,
        min_fft_len: int = 32768,
        safety_sigma: float = 6.0,
        gaussian_sigma: float | np.ndarray | None = None,
        resolving_power: float | None = None,
        template_floor: float = 1e-15,
        attenuation_dtype: str | np.dtype | None = None,
        storage_mode: str = "research",
        isotope_patterns: Mapping[str, IsotopePattern] | None = None,
        isotope_data_version: str | None = None,
    ) -> "CenteredLogPhaseTable":
        """Build a table whose FFT window contains the requested formulas."""

        sizing_table = cls.build(
            elements=elements,
            dm=dm,
            min_fft_len=min_fft_len,
            template_floor=template_floor,
            attenuation_dtype=attenuation_dtype,
            storage_mode=storage_mode,
            isotope_patterns=isotope_patterns,
            isotope_data_version=isotope_data_version,
        )
        n_fft = sizing_table.suggest_n_fft_for_counts(
            counts,
            min_fft_len=min_fft_len,
            safety_sigma=safety_sigma,
            gaussian_sigma=gaussian_sigma,
            resolving_power=resolving_power,
        )
        if n_fft == sizing_table.n_fft:
            return sizing_table
        return cls.build(
            elements=elements,
            dm=dm,
            n_fft=n_fft,
            template_floor=template_floor,
            attenuation_dtype=attenuation_dtype,
            storage_mode=storage_mode,
            isotope_patterns=isotope_patterns,
            isotope_data_version=sizing_table.isotope_data_version,
        )

    @classmethod
    def build_for_formulas(
        cls,
        formulas: Sequence[str | Mapping[str, int]],
        *,
        elements: Sequence[str] = ("C", "H", "N", "O", "S", "Cl", "Br"),
        dm: float = 0.001,
        min_fft_len: int = 32768,
        safety_sigma: float = 6.0,
        gaussian_sigma: float | np.ndarray | None = None,
        resolving_power: float | None = None,
        template_floor: float = 1e-15,
        attenuation_dtype: str | np.dtype | None = None,
        storage_mode: str = "research",
        isotope_patterns: Mapping[str, IsotopePattern] | None = None,
        isotope_data_version: str | None = None,
    ) -> "CenteredLogPhaseTable":
        """Build a table sized for a set of formulas."""

        counts = counts_matrix(formulas, tuple(elements))
        return cls.build_for_counts(
            counts,
            elements=elements,
            dm=dm,
            min_fft_len=min_fft_len,
            safety_sigma=safety_sigma,
            gaussian_sigma=gaussian_sigma,
            resolving_power=resolving_power,
            template_floor=template_floor,
            attenuation_dtype=attenuation_dtype,
            storage_mode=storage_mode,
            isotope_patterns=isotope_patterns,
            isotope_data_version=isotope_data_version,
        )

    @property
    def n_positive(self) -> int:
        return positive_frequency_count(self.n_fft)

    @property
    def cache_key(self) -> TableCacheKey:
        return make_table_cache_key(
            elements=self.elements,
            dm=self.dm,
            n_fft=self.n_fft,
            isotope_data_version=self.isotope_data_version,
        )

    @property
    def has_phase_table(self) -> bool:
        """Return whether double-precision phase tables are retained."""

        return self.phase.shape == self.attenuation.shape

    @property
    def table_nbytes(self) -> int:
        """Return bytes held by the precomputed numeric table arrays."""

        arrays = (
            self.omega,
            self.attenuation,
            self.phase,
            self.phase_cycles,
            self.phase_uint64,
            self.attenuation_count_threshold,
            self.attenuation_template_trigger_count,
            self.mean_masses,
            self.variances,
        )
        return int(sum(array.nbytes for array in arrays))

    def counts_from_formulas(
        self,
        formulas: Sequence[str | Mapping[str, int]],
    ) -> np.ndarray:
        """Parse formulas using this table's element order."""

        return counts_matrix(formulas, self.elements)

    def mean_mass_many_counts(self, counts: np.ndarray) -> np.ndarray:
        """Return neutral mean masses for each count row."""

        counts_2d = self._coerce_counts(counts)
        return counts_2d.astype(np.float64) @ self.mean_masses

    def isotope_variance_many_counts(self, counts: np.ndarray) -> np.ndarray:
        """Return residual isotope-distribution variances for each count row."""

        counts_2d = self._coerce_counts(counts)
        return counts_2d.astype(np.float64) @ self.variances

    def profile_sigma_many_counts(
        self,
        counts: np.ndarray,
        *,
        gaussian_sigma: float | np.ndarray | None = None,
        resolving_power: float | None = None,
    ) -> np.ndarray:
        """Return combined isotope plus optional Gaussian profile sigma."""

        counts_2d = self._coerce_counts(counts)
        variance = self.isotope_variance_many_counts(counts_2d)
        sigma = self._resolve_gaussian_sigma(
            counts_2d,
            gaussian_sigma=gaussian_sigma,
            resolving_power=resolving_power,
        )
        if sigma is not None:
            variance = variance + sigma * sigma
        return np.sqrt(variance)

    def required_window_width_many_counts(
        self,
        counts: np.ndarray,
        *,
        safety_sigma: float = 6.0,
        gaussian_sigma: float | np.ndarray | None = None,
        resolving_power: float | None = None,
    ) -> np.ndarray:
        """Return required full FFT window widths for count rows."""

        if safety_sigma <= 0.0:
            raise ValueError("safety_sigma must be positive")
        profile_sigma = self.profile_sigma_many_counts(
            counts,
            gaussian_sigma=gaussian_sigma,
            resolving_power=resolving_power,
        )
        return 2.0 * float(safety_sigma) * profile_sigma

    def suggest_n_fft_for_counts(
        self,
        counts: np.ndarray,
        *,
        min_fft_len: int | None = None,
        safety_sigma: float = 6.0,
        gaussian_sigma: float | np.ndarray | None = None,
        resolving_power: float | None = None,
    ) -> int:
        """Suggest a fast odd FFT length whose mass window contains profiles."""

        required_width = float(np.max(self.required_window_width_many_counts(
            counts,
            safety_sigma=safety_sigma,
            gaussian_sigma=gaussian_sigma,
            resolving_power=resolving_power,
        )))
        required_len = int(np.ceil(required_width / self.dm))
        lower_bound = max(required_len, self.n_fft if min_fft_len is None else min_fft_len)
        return suggest_fast_odd_len(lower_bound)

    def residual_spectrum_many_counts(
        self,
        counts: np.ndarray,
        *,
        method: str = "log_table",
        gaussian_sigma: float | np.ndarray | None = None,
        resolving_power: float | None = None,
        prune_cutoff: float = 1e-12,
        workers: int | None = None,
    ) -> np.ndarray:
        """Evaluate centered residual characteristic spectra."""

        counts_2d = self._coerce_counts(counts)
        spectrum_workers = _resolve_workers(workers)
        method = self.select_spectrum_method(
            counts_2d,
            method=method,
            gaussian_sigma=gaussian_sigma,
            resolving_power=resolving_power,
            prune_cutoff=prune_cutoff,
        )
        if method == "log_pruned":
            return self._log_pruned_residual_spectrum_many_counts(
                counts_2d,
                gaussian_sigma=gaussian_sigma,
                resolving_power=resolving_power,
                prune_cutoff=prune_cutoff,
            )
        if method == "cython_log_pruned":
            return self._cython_log_pruned_residual_spectrum_many_counts(
                counts_2d,
                gaussian_sigma=gaussian_sigma,
                resolving_power=resolving_power,
                prune_cutoff=prune_cutoff,
                workers=spectrum_workers,
            )
        if method == "cython_log_pruned_modphase":
            return self._cython_log_pruned_modphase_residual_spectrum_many_counts(
                counts_2d,
                gaussian_sigma=gaussian_sigma,
                resolving_power=resolving_power,
                prune_cutoff=prune_cutoff,
                workers=spectrum_workers,
            )
        if method == "cython_log_pruned_cyclephase":
            return self._cython_log_pruned_cyclephase_residual_spectrum_many_counts(
                counts_2d,
                gaussian_sigma=gaussian_sigma,
                resolving_power=resolving_power,
                prune_cutoff=prune_cutoff,
                workers=spectrum_workers,
            )
        if method == "cython_log_pruned_uintphase":
            return self._cython_log_pruned_uintphase_residual_spectrum_many_counts(
                counts_2d,
                gaussian_sigma=gaussian_sigma,
                resolving_power=resolving_power,
                prune_cutoff=prune_cutoff,
                workers=spectrum_workers,
            )
        if method == "cython_log_pruned_uintphase_threshold":
            return self._cython_log_pruned_uintphase_threshold_residual_spectrum_many_counts(
                counts_2d,
                gaussian_sigma=gaussian_sigma,
                resolving_power=resolving_power,
                prune_cutoff=prune_cutoff,
                workers=spectrum_workers,
            )
        if method == "cython_log_pruned_attn32":
            return self._cython_log_pruned_attn32_residual_spectrum_many_counts(
                counts_2d,
                gaussian_sigma=gaussian_sigma,
                resolving_power=resolving_power,
                prune_cutoff=prune_cutoff,
                workers=spectrum_workers,
            )
        if method == "cython_log_pruned_attn32_uintphase":
            return self._cython_log_pruned_attn32_uintphase_residual_spectrum_many_counts(
                counts_2d,
                gaussian_sigma=gaussian_sigma,
                resolving_power=resolving_power,
                prune_cutoff=prune_cutoff,
                workers=spectrum_workers,
            )
        if method == "cython_log_pruned_attn32_uintphase_threshold":
            return self._cython_log_pruned_attn32_uintphase_threshold_residual_spectrum_many_counts(
                counts_2d,
                gaussian_sigma=gaussian_sigma,
                resolving_power=resolving_power,
                prune_cutoff=prune_cutoff,
                workers=spectrum_workers,
            )
        if method == "log_table":
            spectrum = self._log_table_residual_spectrum_many_counts(counts_2d)
        elif method == "cython_log_table":
            spectrum = self._cython_log_table_residual_spectrum_many_counts(counts_2d)
        elif method == "direct_rebuild":
            spectrum = self._direct_rebuild_residual_spectrum_many_counts(counts_2d)
        else:
            raise ValueError(f"unknown spectrum evaluation method {method!r}")
        return self._apply_gaussian_damping(
            spectrum,
            counts_2d,
            gaussian_sigma=gaussian_sigma,
            resolving_power=resolving_power,
        )

    def select_spectrum_method(
        self,
        counts: np.ndarray,
        *,
        method: str = "cython_auto",
        gaussian_sigma: float | np.ndarray | None = None,
        resolving_power: float | None = None,
        prune_cutoff: float = 1e-12,
        active_fraction_threshold: float = 0.1,
        template_active_fraction_threshold: float = 0.25,
    ) -> str:
        """Resolve an automatic spectrum method for a count matrix."""

        if method not in {"auto", "cython_auto"}:
            return method
        if _cython_log_table is None:
            if not self.has_phase_table:
                raise RuntimeError("production storage requires the Cython backend")
            return "log_pruned"
        if prune_cutoff != self.threshold_prune_cutoff:
            return self._default_cython_pruned_method()
        sigma = self._resolve_gaussian_sigma(
            counts,
            gaussian_sigma=gaussian_sigma,
            resolving_power=resolving_power,
        )
        if sigma is not None and np.any(sigma > 0.0):
            return self._default_cython_pruned_method()

        trigger_count = self._template_trigger_count(template_active_fraction_threshold)
        counts_2d = self._coerce_counts(counts)
        active_triggers = trigger_count > 0
        if np.any(active_triggers):
            trigger_hits = counts_2d[:, active_triggers] >= trigger_count[active_triggers]
            if np.any(trigger_hits):
                return self._threshold_cython_pruned_method()
        return self._default_cython_pruned_method()

    def _default_cython_pruned_method(self) -> str:
        if self.attenuation.dtype == np.dtype(np.float32):
            if not self.has_phase_table:
                return "cython_log_pruned_attn32_uintphase"
            return "cython_log_pruned_attn32"
        if self.has_phase_table:
            return "cython_log_pruned"
        return "cython_log_pruned_uintphase"

    def _threshold_cython_pruned_method(self) -> str:
        if self.attenuation.dtype == np.dtype(np.float32):
            return "cython_log_pruned_attn32_uintphase_threshold"
        return "cython_log_pruned_uintphase_threshold"

    def _template_trigger_count(
        self,
        active_fraction_threshold: float,
    ) -> np.ndarray:
        if active_fraction_threshold == 0.25:
            return self.attenuation_template_trigger_count
        return _attenuation_template_trigger_count(
            self.attenuation_count_threshold,
            active_fraction_threshold=active_fraction_threshold,
        )

    def active_frequency_fraction(
        self,
        counts: np.ndarray,
        *,
        amplitude_cutoff: float = 1e-12,
        gaussian_sigma: float | np.ndarray | None = None,
        resolving_power: float | None = None,
    ) -> np.ndarray:
        """Return the fraction of positive-frequency bins above a cutoff.

        This is a cheap planning metric for future log-pruned evaluation. Large
        molecules usually have a smaller active fraction because attenuation
        accumulates across atom counts.
        """

        if not (0.0 < amplitude_cutoff < 1.0):
            raise ValueError("amplitude_cutoff must be between 0 and 1")
        counts_2d = self._coerce_counts(counts)
        total_attenuation = counts_2d.astype(np.float64) @ self.attenuation
        sigma = self._resolve_gaussian_sigma(
            counts_2d,
            gaussian_sigma=gaussian_sigma,
            resolving_power=resolving_power,
        )
        if sigma is not None:
            total_attenuation = total_attenuation + _gaussian_attenuation(
                sigma,
                self.omega,
            )
        max_attenuation = -np.log(amplitude_cutoff)
        return np.mean(total_attenuation <= max_attenuation, axis=-1)

    def mass_profile_many_counts(
        self,
        counts: np.ndarray,
        *,
        method: str = "log_table",
        gaussian_sigma: float | np.ndarray | None = None,
        resolving_power: float | None = None,
        workers: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray | float | int | str]]:
        """Return shifted mass axes and residual profiles for count rows."""

        counts_2d = self._coerce_counts(counts)
        selected_method = self.select_spectrum_method(
            counts_2d,
            method=method,
            gaussian_sigma=gaussian_sigma,
            resolving_power=resolving_power,
        )
        spectrum = self.residual_spectrum_many_counts(
            counts_2d,
            method=selected_method,
            gaussian_sigma=gaussian_sigma,
            resolving_power=resolving_power,
            workers=workers,
        )
        profiles = fast_odd_irfft(
            spectrum,
            n=self.n_fft,
            axis=-1,
            workers=workers,
        )
        profiles = np.fft.fftshift(profiles.real, axes=-1)
        profiles = profiles[:, ::-1]
        residual_axis = self.residual_mass_axis()
        mean_masses = self.mean_mass_many_counts(counts_2d)
        mass_axis = mean_masses[:, None] + residual_axis[None, :]
        info: dict[str, np.ndarray | float | int | str] = {
            "method": selected_method,
            "dm": self.dm,
            "n_fft": self.n_fft,
            "mean_masses": mean_masses,
            "residual_axis": residual_axis,
        }
        if selected_method != method:
            info["requested_method"] = method
        if resolving_power is not None:
            info["resolving_power"] = float(resolving_power)
        if workers is not None:
            info["workers"] = _resolve_workers(workers)
        return mass_axis, profiles, info

    def mass_profile_window_many_counts(
        self,
        counts: np.ndarray,
        *,
        residual_start: float | None = None,
        residual_stop: float | None = None,
        mass_start: float | None = None,
        mass_stop: float | None = None,
        output_dm: float | None = None,
        n_points: int | None = None,
        method: str = "log_table",
        gaussian_sigma: float | np.ndarray | None = None,
        resolving_power: float | None = None,
        prune_cutoff: float = 1e-12,
        workers: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray | float | int | str]]:
        """Return a CZT-evaluated mass-profile window.

        Pass either a residual window, relative to each formula's mean mass, or
        an absolute mass window. ``output_dm`` controls the requested output
        spacing and can differ from the table spacing used to sample the
        characteristic function. Alternatively, pass ``n_points`` to sample the
        inclusive start/stop interval with ``np.linspace``.
        """

        counts_2d = self._coerce_counts(counts)
        residual_mode = residual_start is not None or residual_stop is not None
        mass_mode = mass_start is not None or mass_stop is not None
        if residual_mode == mass_mode:
            raise ValueError(
                "Pass either residual_start/residual_stop or mass_start/mass_stop"
            )
        if residual_mode:
            if residual_start is None or residual_stop is None:
                raise ValueError("residual_start and residual_stop must be passed together")
            output_axis = _regular_window_axis(
                float(residual_start),
                float(residual_stop),
                output_dm=output_dm,
                n_points=n_points,
            )
        else:
            if mass_start is None or mass_stop is None:
                raise ValueError("mass_start and mass_stop must be passed together")
            output_axis = _regular_window_axis(
                float(mass_start),
                float(mass_stop),
                output_dm=output_dm,
                n_points=n_points,
            )

        selected_method = self.select_spectrum_method(
            counts_2d,
            method=method,
            gaussian_sigma=gaussian_sigma,
            resolving_power=resolving_power,
            prune_cutoff=prune_cutoff,
        )
        spectrum = self.residual_spectrum_many_counts(
            counts_2d,
            method=selected_method,
            gaussian_sigma=gaussian_sigma,
            resolving_power=resolving_power,
            prune_cutoff=prune_cutoff,
            workers=workers,
        )
        mean_masses = self.mean_mass_many_counts(counts_2d)

        if output_axis.shape[0] == 1:
            axis_step = self.dm
        else:
            axis_step = float(output_axis[1] - output_axis[0])
        czt_step = -axis_step / self.dm

        if residual_mode:
            czt_start = -float(output_axis[0]) / self.dm
            profiles = czt_irfft_window(
                spectrum,
                n=self.n_fft,
                start=czt_start,
                step=czt_step,
                m=output_axis.shape[0],
                axis=-1,
                trim_zeros=True,
            )
            mass_axis = mean_masses[:, None] + output_axis[None, :]
            residual_axis = output_axis
        else:
            rows = []
            for row_idx, mean_mass in enumerate(mean_masses):
                czt_start = -(float(output_axis[0]) - float(mean_mass)) / self.dm
                row = czt_irfft_window(
                    spectrum[row_idx],
                    n=self.n_fft,
                    start=czt_start,
                    step=czt_step,
                    m=output_axis.shape[0],
                    axis=-1,
                    trim_zeros=True,
                )
                rows.append(row)
            profiles = np.stack(rows)
            mass_axis = np.broadcast_to(output_axis, profiles.shape)
            residual_axis = mass_axis - mean_masses[:, None]

        info: dict[str, np.ndarray | float | int | str] = {
            "method": selected_method,
            "transform": "czt_irfft_window",
            "dm": self.dm,
            "output_dm": axis_step,
            "n_fft": self.n_fft,
            "n_points": output_axis.shape[0],
            "mean_masses": mean_masses,
            "residual_axis": residual_axis,
        }
        if selected_method != method:
            info["requested_method"] = method
        if resolving_power is not None:
            info["resolving_power"] = float(resolving_power)
        if workers is not None:
            info["workers"] = _resolve_workers(workers)
        return mass_axis, profiles, info

    def residual_mass_axis(self) -> np.ndarray:
        """Return the fft-shifted residual mass axis."""

        return (np.arange(self.n_fft, dtype=np.float64) - self.n_fft // 2) * self.dm

    def _require_phase_table(self, method: str) -> None:
        if not self.has_phase_table:
            raise RuntimeError(
                f"{method} requires research storage with retained phase tables"
            )

    def _log_table_residual_spectrum_many_counts(self, counts: np.ndarray) -> np.ndarray:
        self._require_phase_table("log_table")
        counts_float = counts.astype(np.float64, copy=False)
        attenuation = counts_float @ self.attenuation
        phase = counts_float @ self.phase
        magnitude = np.exp(-attenuation)
        return magnitude * (np.cos(phase) + 1j * np.sin(phase))

    def _log_pruned_residual_spectrum_many_counts(
        self,
        counts: np.ndarray,
        *,
        gaussian_sigma: float | np.ndarray | None,
        resolving_power: float | None,
        prune_cutoff: float,
    ) -> np.ndarray:
        self._require_phase_table("log_pruned")
        if not (0.0 < prune_cutoff < 1.0):
            raise ValueError("prune_cutoff must be between 0 and 1")

        counts_float = counts.astype(np.float64, copy=False)
        attenuation = counts_float @ self.attenuation
        sigma = self._resolve_gaussian_sigma(
            counts,
            gaussian_sigma=gaussian_sigma,
            resolving_power=resolving_power,
        )
        if sigma is not None:
            attenuation = attenuation + _gaussian_attenuation(sigma, self.omega)

        max_attenuation = -np.log(prune_cutoff)
        spectra = np.zeros((counts.shape[0], self.n_positive), dtype=np.complex128)
        for row_idx, row in enumerate(counts_float):
            active = attenuation[row_idx] <= max_attenuation
            if not np.any(active):
                continue
            phase = row @ self.phase[:, active]
            magnitude = np.exp(-attenuation[row_idx, active])
            spectra[row_idx, active] = magnitude * (
                np.cos(phase) + 1j * np.sin(phase)
            )
        return spectra

    def _cython_log_table_residual_spectrum_many_counts(
        self,
        counts: np.ndarray,
    ) -> np.ndarray:
        self._require_phase_table("cython_log_table")
        if _cython_log_table is None:
            raise RuntimeError("Cython backend is not built")
        return _cython_log_table.log_table_spectrum(
            np.ascontiguousarray(counts, dtype=np.int64),
            np.ascontiguousarray(self.attenuation, dtype=np.float64),
            np.ascontiguousarray(self.phase, dtype=np.float64),
        )

    def _cython_log_pruned_residual_spectrum_many_counts(
        self,
        counts: np.ndarray,
        *,
        gaussian_sigma: float | np.ndarray | None,
        resolving_power: float | None,
        prune_cutoff: float,
        workers: int = 1,
    ) -> np.ndarray:
        self._require_phase_table("cython_log_pruned")
        if _cython_log_table is None:
            raise RuntimeError("Cython backend is not built")
        sigma = self._resolve_gaussian_sigma(
            counts,
            gaussian_sigma=gaussian_sigma,
            resolving_power=resolving_power,
        )
        if sigma is None:
            sigma = np.empty(0, dtype=np.float64)
        else:
            sigma = np.ascontiguousarray(sigma, dtype=np.float64)
        return _cython_log_table.log_pruned_spectrum(
            np.ascontiguousarray(counts, dtype=np.int64),
            np.ascontiguousarray(self.attenuation, dtype=np.float64),
            np.ascontiguousarray(self.phase, dtype=np.float64),
            np.ascontiguousarray(self.omega, dtype=np.float64),
            sigma,
            float(prune_cutoff),
            int(workers),
        )

    def _cython_log_pruned_modphase_residual_spectrum_many_counts(
        self,
        counts: np.ndarray,
        *,
        gaussian_sigma: float | np.ndarray | None,
        resolving_power: float | None,
        prune_cutoff: float,
        workers: int = 1,
    ) -> np.ndarray:
        self._require_phase_table("cython_log_pruned_modphase")
        if _cython_log_table is None:
            raise RuntimeError("Cython backend is not built")
        sigma = self._resolve_gaussian_sigma(
            counts,
            gaussian_sigma=gaussian_sigma,
            resolving_power=resolving_power,
        )
        if sigma is None:
            sigma = np.empty(0, dtype=np.float64)
        else:
            sigma = np.ascontiguousarray(sigma, dtype=np.float64)
        return _cython_log_table.log_pruned_spectrum_modphase(
            np.ascontiguousarray(counts, dtype=np.int64),
            np.ascontiguousarray(self.attenuation, dtype=np.float64),
            np.ascontiguousarray(self.phase, dtype=np.float64),
            np.ascontiguousarray(self.omega, dtype=np.float64),
            sigma,
            float(prune_cutoff),
        )

    def _cython_log_pruned_cyclephase_residual_spectrum_many_counts(
        self,
        counts: np.ndarray,
        *,
        gaussian_sigma: float | np.ndarray | None,
        resolving_power: float | None,
        prune_cutoff: float,
        workers: int = 1,
    ) -> np.ndarray:
        self._require_phase_table("cython_log_pruned_cyclephase")
        if _cython_log_table is None:
            raise RuntimeError("Cython backend is not built")
        sigma = self._resolve_gaussian_sigma(
            counts,
            gaussian_sigma=gaussian_sigma,
            resolving_power=resolving_power,
        )
        if sigma is None:
            sigma = np.empty(0, dtype=np.float64)
        else:
            sigma = np.ascontiguousarray(sigma, dtype=np.float64)
        return _cython_log_table.log_pruned_spectrum_cyclephase(
            np.ascontiguousarray(counts, dtype=np.int64),
            np.ascontiguousarray(self.attenuation, dtype=np.float64),
            np.ascontiguousarray(self.phase_cycles, dtype=np.float64),
            np.ascontiguousarray(self.omega, dtype=np.float64),
            sigma,
            float(prune_cutoff),
        )

    def _cython_log_pruned_uintphase_residual_spectrum_many_counts(
        self,
        counts: np.ndarray,
        *,
        gaussian_sigma: float | np.ndarray | None,
        resolving_power: float | None,
        prune_cutoff: float,
        workers: int = 1,
    ) -> np.ndarray:
        if _cython_log_table is None:
            raise RuntimeError("Cython backend is not built")
        sigma = self._resolve_gaussian_sigma(
            counts,
            gaussian_sigma=gaussian_sigma,
            resolving_power=resolving_power,
        )
        if sigma is None:
            sigma = np.empty(0, dtype=np.float64)
        else:
            sigma = np.ascontiguousarray(sigma, dtype=np.float64)
        return _cython_log_table.log_pruned_spectrum_uintphase(
            np.ascontiguousarray(counts, dtype=np.int64),
            np.ascontiguousarray(self.attenuation, dtype=np.float64),
            np.ascontiguousarray(self.phase_uint64, dtype=np.uint64),
            np.ascontiguousarray(self.omega, dtype=np.float64),
            sigma,
            float(prune_cutoff),
            int(workers),
        )

    def _cython_log_pruned_uintphase_threshold_residual_spectrum_many_counts(
        self,
        counts: np.ndarray,
        *,
        gaussian_sigma: float | np.ndarray | None,
        resolving_power: float | None,
        prune_cutoff: float,
        workers: int = 1,
    ) -> np.ndarray:
        if _cython_log_table is None:
            raise RuntimeError("Cython backend is not built")
        if prune_cutoff != self.threshold_prune_cutoff:
            raise ValueError(
                "cython_log_pruned_uintphase_threshold requires "
                f"prune_cutoff={self.threshold_prune_cutoff:g}"
            )
        sigma = self._resolve_gaussian_sigma(
            counts,
            gaussian_sigma=gaussian_sigma,
            resolving_power=resolving_power,
        )
        if sigma is None:
            sigma = np.empty(0, dtype=np.float64)
        else:
            sigma = np.ascontiguousarray(sigma, dtype=np.float64)
        return _cython_log_table.log_pruned_spectrum_uintphase_threshold(
            np.ascontiguousarray(counts, dtype=np.int64),
            np.ascontiguousarray(self.attenuation, dtype=np.float64),
            np.ascontiguousarray(self.phase_uint64, dtype=np.uint64),
            np.ascontiguousarray(self.attenuation_count_threshold, dtype=np.uint32),
            np.ascontiguousarray(self.omega, dtype=np.float64),
            sigma,
            float(prune_cutoff),
            int(workers),
        )

    def _cython_log_pruned_attn32_residual_spectrum_many_counts(
        self,
        counts: np.ndarray,
        *,
        gaussian_sigma: float | np.ndarray | None,
        resolving_power: float | None,
        prune_cutoff: float,
        workers: int = 1,
    ) -> np.ndarray:
        self._require_phase_table("cython_log_pruned_attn32")
        if _cython_log_table is None:
            raise RuntimeError("Cython backend is not built")
        sigma = self._resolve_gaussian_sigma(
            counts,
            gaussian_sigma=gaussian_sigma,
            resolving_power=resolving_power,
        )
        if sigma is None:
            sigma = np.empty(0, dtype=np.float64)
        else:
            sigma = np.ascontiguousarray(sigma, dtype=np.float64)
        return _cython_log_table.log_pruned_spectrum_attenuation32(
            np.ascontiguousarray(counts, dtype=np.int64),
            np.ascontiguousarray(self.attenuation, dtype=np.float32),
            np.ascontiguousarray(self.phase, dtype=np.float64),
            np.ascontiguousarray(self.omega, dtype=np.float64),
            sigma,
            float(prune_cutoff),
            int(workers),
        )

    def _cython_log_pruned_attn32_uintphase_residual_spectrum_many_counts(
        self,
        counts: np.ndarray,
        *,
        gaussian_sigma: float | np.ndarray | None,
        resolving_power: float | None,
        prune_cutoff: float,
        workers: int = 1,
    ) -> np.ndarray:
        if _cython_log_table is None:
            raise RuntimeError("Cython backend is not built")
        sigma = self._resolve_gaussian_sigma(
            counts,
            gaussian_sigma=gaussian_sigma,
            resolving_power=resolving_power,
        )
        if sigma is None:
            sigma = np.empty(0, dtype=np.float64)
        else:
            sigma = np.ascontiguousarray(sigma, dtype=np.float64)
        return _cython_log_table.log_pruned_spectrum_attenuation32_uintphase(
            np.ascontiguousarray(counts, dtype=np.int64),
            np.ascontiguousarray(self.attenuation, dtype=np.float32),
            np.ascontiguousarray(self.phase_uint64, dtype=np.uint64),
            np.ascontiguousarray(self.omega, dtype=np.float64),
            sigma,
            float(prune_cutoff),
            int(workers),
        )

    def _cython_log_pruned_attn32_uintphase_threshold_residual_spectrum_many_counts(
        self,
        counts: np.ndarray,
        *,
        gaussian_sigma: float | np.ndarray | None,
        resolving_power: float | None,
        prune_cutoff: float,
        workers: int = 1,
    ) -> np.ndarray:
        if _cython_log_table is None:
            raise RuntimeError("Cython backend is not built")
        if prune_cutoff != self.threshold_prune_cutoff:
            raise ValueError(
                "cython_log_pruned_attn32_uintphase_threshold requires "
                f"prune_cutoff={self.threshold_prune_cutoff:g}"
            )
        sigma = self._resolve_gaussian_sigma(
            counts,
            gaussian_sigma=gaussian_sigma,
            resolving_power=resolving_power,
        )
        if sigma is None:
            sigma = np.empty(0, dtype=np.float64)
        else:
            sigma = np.ascontiguousarray(sigma, dtype=np.float64)
        return _cython_log_table.log_pruned_spectrum_attenuation32_uintphase_threshold(
            np.ascontiguousarray(counts, dtype=np.int64),
            np.ascontiguousarray(self.attenuation, dtype=np.float32),
            np.ascontiguousarray(self.phase_uint64, dtype=np.uint64),
            np.ascontiguousarray(self.attenuation_count_threshold, dtype=np.uint32),
            np.ascontiguousarray(self.omega, dtype=np.float64),
            sigma,
            float(prune_cutoff),
            int(workers),
        )

    def _direct_rebuild_residual_spectrum_many_counts(self, counts: np.ndarray) -> np.ndarray:
        spectra = np.ones((counts.shape[0], self.n_positive), dtype=np.complex128)
        for row_idx, row in enumerate(counts):
            spectrum = spectra[row_idx]
            for element_idx, atom_count in enumerate(row):
                if atom_count == 0:
                    continue
                char = _direct_element_characteristic(
                    self.isotope_patterns[element_idx],
                    self.omega,
                )
                spectrum *= char ** int(atom_count)
        return spectra

    def _apply_gaussian_damping(
        self,
        spectrum: np.ndarray,
        counts: np.ndarray,
        *,
        gaussian_sigma: float | np.ndarray | None,
        resolving_power: float | None,
    ) -> np.ndarray:
        sigma = self._resolve_gaussian_sigma(
            counts,
            gaussian_sigma=gaussian_sigma,
            resolving_power=resolving_power,
        )
        if sigma is None:
            return spectrum
        return spectrum * np.exp(-_gaussian_attenuation(sigma, self.omega))

    def _resolve_gaussian_sigma(
        self,
        counts: np.ndarray,
        *,
        gaussian_sigma: float | np.ndarray | None,
        resolving_power: float | None,
    ) -> np.ndarray | None:
        if gaussian_sigma is not None and resolving_power is not None:
            raise ValueError("Pass either gaussian_sigma or resolving_power, not both")
        if resolving_power is not None:
            if resolving_power <= 0.0:
                raise ValueError("resolving_power must be positive")
            fwhm = self.mean_mass_many_counts(counts) / float(resolving_power)
            return fwhm / (2.0 * sqrt(2.0 * log(2.0)))
        if gaussian_sigma is None:
            return None

        sigma = np.asarray(gaussian_sigma, dtype=np.float64)
        if sigma.ndim == 0:
            sigma = np.full(counts.shape[0], float(sigma), dtype=np.float64)
        if sigma.shape != (counts.shape[0],):
            raise ValueError(
                "gaussian_sigma must be a scalar or one value per formula"
            )
        if np.any(sigma < 0.0):
            raise ValueError("gaussian_sigma must be non-negative")
        return sigma

    def _coerce_counts(self, counts: np.ndarray) -> np.ndarray:
        counts_array = np.asarray(counts)
        if counts_array.ndim == 1:
            counts_array = counts_array[None, :]
        if counts_array.ndim != 2 or counts_array.shape[1] != len(self.elements):
            raise ValueError(
                "counts must have shape "
                f"(n_formulas, {len(self.elements)}) or ({len(self.elements)},)"
            )
        if np.any(counts_array < 0):
            raise ValueError("counts must be non-negative")
        return counts_array.astype(np.int64, copy=False)


def _build_element_log_phase(
    pattern: IsotopePattern,
    omega: np.ndarray,
    *,
    template_floor: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    mean_mass = pattern.mean_mass
    deltas = pattern.masses - mean_mass
    angles = np.multiply.outer(deltas, omega)
    real = pattern.abundances @ np.cos(angles)
    imag = pattern.abundances @ np.sin(angles)
    magnitude = np.hypot(real, imag)
    attenuation = -np.log(np.maximum(magnitude, template_floor))
    phase = np.unwrap(np.arctan2(imag, real))
    attenuation[0] = 0.0
    phase[0] = 0.0
    return attenuation, phase, mean_mass


def _wrap_phase_cycles(phase: np.ndarray) -> np.ndarray:
    cycles = phase / (2.0 * pi)
    return cycles - np.floor(cycles)


def _phase_cycles_to_uint64(phase_cycles: np.ndarray) -> np.ndarray:
    cycles = np.minimum(phase_cycles, np.nextafter(1.0, 0.0))
    scaled_hi = cycles * 2.0**32
    hi = np.floor(scaled_hi).astype(np.uint64)
    lo_fraction = scaled_hi - hi.astype(np.float64)
    lo = np.floor(lo_fraction * 2.0**32).astype(np.uint64)
    return (hi << np.uint64(32)) | lo


def _attenuation_count_threshold(
    attenuation: np.ndarray,
    *,
    prune_cutoff: float,
) -> np.ndarray:
    if not (0.0 < prune_cutoff < 1.0):
        raise ValueError("prune_cutoff must be between 0 and 1")
    max_attenuation = -log(prune_cutoff)
    max_uint32 = np.iinfo(np.uint32).max
    thresholds = np.zeros(attenuation.shape, dtype=np.uint32)
    min_representable = max_attenuation / float(max_uint32)
    active = attenuation > min_representable
    values = np.floor(max_attenuation / attenuation[active]) + 1.0
    representable = values <= float(max_uint32)
    active_indices = np.flatnonzero(active)
    thresholds.flat[active_indices[representable]] = values[representable].astype(np.uint32)
    return thresholds


def _attenuation_template_trigger_count(
    count_threshold: np.ndarray,
    *,
    active_fraction_threshold: float,
) -> np.ndarray:
    if not (0.0 < active_fraction_threshold < 1.0):
        raise ValueError("active_fraction_threshold must be between 0 and 1")
    target_pruned = int(np.ceil((1.0 - active_fraction_threshold) * count_threshold.shape[1]))
    triggers = np.zeros(count_threshold.shape[0], dtype=np.uint32)
    for element_idx, row in enumerate(count_threshold):
        finite = row[row != 0]
        if finite.shape[0] < target_pruned:
            continue
        kth = target_pruned - 1
        triggers[element_idx] = np.partition(finite, kth)[kth]
    return triggers


def _direct_element_characteristic(
    pattern: IsotopePattern,
    omega: np.ndarray,
) -> np.ndarray:
    deltas = pattern.masses - pattern.mean_mass
    angles = np.multiply.outer(deltas, omega)
    real = pattern.abundances @ np.cos(angles)
    imag = pattern.abundances @ np.sin(angles)
    return real + 1j * imag


def _gaussian_attenuation(sigma: np.ndarray, omega: np.ndarray) -> np.ndarray:
    return 0.5 * (sigma[:, None] * omega[None, :]) ** 2


def _regular_window_axis(
    start: float,
    stop: float,
    *,
    output_dm: float | None,
    n_points: int | None,
) -> np.ndarray:
    if stop <= start:
        raise ValueError("window stop must be greater than start")
    if output_dm is not None and n_points is not None:
        raise ValueError("Pass either output_dm or n_points, not both")
    if n_points is not None:
        n_points = int(n_points)
        if n_points < 1:
            raise ValueError("n_points must be positive")
        return np.linspace(start, stop, n_points, dtype=np.float64)
    if output_dm is None:
        raise ValueError("Pass output_dm or n_points")
    output_dm = float(output_dm)
    if output_dm <= 0.0:
        raise ValueError("output_dm must be positive")
    count = int(np.floor((stop - start) / output_dm + 0.5)) + 1
    axis = start + output_dm * np.arange(count, dtype=np.float64)
    return axis[axis <= stop + 0.5 * output_dm]


def _resolve_workers(workers: int | None) -> int:
    if workers is None:
        return 1
    workers = int(workers)
    if workers < 1:
        raise ValueError("workers must be positive")
    return workers
