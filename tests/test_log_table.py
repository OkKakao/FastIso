import numpy as np
import pytest

from fastiso import CenteredLogPhaseTable, has_cython_backend


def test_table_builds_expected_shapes():
    table = CenteredLogPhaseTable.build(
        elements=("C", "H", "N", "O", "S"),
        dm=0.01,
        min_fft_len=2048,
    )

    assert table.n_fft % 2 == 1
    assert table.attenuation.shape == (5, table.n_positive)
    assert table.attenuation.dtype == np.float64
    assert table.phase.shape == (5, table.n_positive)
    assert table.phase_cycles.shape == (5, table.n_positive)
    assert table.phase_uint64.shape == (5, table.n_positive)
    assert table.phase_uint64.dtype == np.uint64
    assert table.attenuation_count_threshold.shape == (5, table.n_positive)
    assert table.attenuation_count_threshold.dtype == np.uint32
    assert table.attenuation_template_trigger_count.shape == (5,)
    assert table.attenuation_template_trigger_count.dtype == np.uint32
    assert np.all((0.0 <= table.phase_cycles) & (table.phase_cycles < 1.0))
    assert table.omega.shape == (table.n_positive,)
    assert table.variances.shape == (5,)
    assert table.storage_mode == "research"
    assert table.has_phase_table
    assert table.table_nbytes > 0


def test_log_table_matches_direct_rebuild_for_moderate_formula():
    table = CenteredLogPhaseTable.build(
        elements=("C", "H", "N", "O", "S"),
        dm=0.01,
        min_fft_len=2048,
    )
    counts = table.counts_from_formulas(["C10H16O2"])

    log_spectrum = table.residual_spectrum_many_counts(counts)
    direct_spectrum = table.residual_spectrum_many_counts(
        counts,
        method="direct_rebuild",
    )

    assert _relative_l2(log_spectrum, direct_spectrum) < 1e-12


def test_log_table_matches_direct_rebuild_for_large_formula():
    table = CenteredLogPhaseTable.build(
        elements=("C", "H", "N", "O", "S"),
        dm=0.01,
        min_fft_len=2048,
    )
    counts = table.counts_from_formulas(["C500H800N125O200S10"])

    log_spectrum = table.residual_spectrum_many_counts(counts)
    direct_spectrum = table.residual_spectrum_many_counts(
        counts,
        method="direct_rebuild",
    )

    assert _relative_l2(log_spectrum, direct_spectrum) < 1e-10


def test_log_table_matches_direct_rebuild_with_gaussian_damping():
    table = CenteredLogPhaseTable.build(
        elements=("C", "H", "N", "O", "S"),
        dm=0.01,
        min_fft_len=2048,
    )
    counts = table.counts_from_formulas(["C100H160N25O40S2"])

    log_spectrum = table.residual_spectrum_many_counts(
        counts,
        resolving_power=100_000,
    )
    direct_spectrum = table.residual_spectrum_many_counts(
        counts,
        method="direct_rebuild",
        resolving_power=100_000,
    )

    assert _relative_l2(log_spectrum, direct_spectrum) < 1e-12


def test_log_pruned_matches_full_log_table_with_gaussian_damping():
    table = CenteredLogPhaseTable.build(
        elements=("C", "H", "N", "O", "S"),
        dm=0.01,
        min_fft_len=2048,
    )
    counts = table.counts_from_formulas(["C500H800N125O200S10"])

    full_spectrum = table.residual_spectrum_many_counts(
        counts,
        resolving_power=100_000,
    )
    pruned_spectrum = table.residual_spectrum_many_counts(
        counts,
        method="log_pruned",
        resolving_power=100_000,
    )

    assert _relative_l2(pruned_spectrum, full_spectrum) < 1e-10


@pytest.mark.skipif(not has_cython_backend(), reason="Cython backend is not built")
def test_cython_log_table_matches_python_log_table():
    table = CenteredLogPhaseTable.build(
        elements=("C", "H", "N", "O", "S"),
        dm=0.01,
        min_fft_len=2048,
    )
    counts = table.counts_from_formulas(["C10H16O2", "C500H800N125O200S10"])

    python_spectrum = table.residual_spectrum_many_counts(counts)
    cython_spectrum = table.residual_spectrum_many_counts(
        counts,
        method="cython_log_table",
    )

    assert _relative_l2(cython_spectrum, python_spectrum) < 1e-12


@pytest.mark.skipif(not has_cython_backend(), reason="Cython backend is not built")
def test_cython_log_pruned_matches_python_log_pruned():
    table = CenteredLogPhaseTable.build(
        elements=("C", "H", "N", "O", "S"),
        dm=0.01,
        min_fft_len=2048,
    )
    counts = table.counts_from_formulas(["C500H800N125O200S10"])

    python_spectrum = table.residual_spectrum_many_counts(
        counts,
        method="log_pruned",
        resolving_power=100_000,
    )
    cython_spectrum = table.residual_spectrum_many_counts(
        counts,
        method="cython_log_pruned",
        resolving_power=100_000,
    )

    assert _relative_l2(cython_spectrum, python_spectrum) < 1e-12


@pytest.mark.skipif(not has_cython_backend(), reason="Cython backend is not built")
def test_cython_log_pruned_modphase_matches_cython_log_pruned():
    table = CenteredLogPhaseTable.build_for_formulas(
        ["C10000H16000N2600O3600S200"],
        elements=("C", "H", "N", "O", "S"),
        dm=0.002,
        min_fft_len=32768,
        resolving_power=100_000,
    )
    counts = table.counts_from_formulas(["C10000H16000N2600O3600S200"])

    baseline = table.residual_spectrum_many_counts(
        counts,
        method="cython_log_pruned",
        resolving_power=100_000,
    )
    modphase = table.residual_spectrum_many_counts(
        counts,
        method="cython_log_pruned_modphase",
        resolving_power=100_000,
    )

    assert _relative_l2(modphase, baseline) < 1e-12


@pytest.mark.skipif(not has_cython_backend(), reason="Cython backend is not built")
def test_cython_log_pruned_cyclephase_matches_cython_log_pruned():
    table = CenteredLogPhaseTable.build_for_formulas(
        ["C10000H16000N2600O3600S200"],
        elements=("C", "H", "N", "O", "S"),
        dm=0.002,
        min_fft_len=32768,
        resolving_power=100_000,
    )
    counts = table.counts_from_formulas(["C10000H16000N2600O3600S200"])

    baseline = table.residual_spectrum_many_counts(
        counts,
        method="cython_log_pruned",
        resolving_power=100_000,
    )
    cyclephase = table.residual_spectrum_many_counts(
        counts,
        method="cython_log_pruned_cyclephase",
        resolving_power=100_000,
    )

    assert _relative_l2(cyclephase, baseline) < 1e-10


@pytest.mark.skipif(not has_cython_backend(), reason="Cython backend is not built")
def test_cython_log_pruned_uintphase_matches_cython_log_pruned():
    table = CenteredLogPhaseTable.build_for_formulas(
        ["C10000H16000N2600O3600S200"],
        elements=("C", "H", "N", "O", "S"),
        dm=0.002,
        min_fft_len=32768,
        resolving_power=100_000,
    )
    counts = table.counts_from_formulas(["C10000H16000N2600O3600S200"])

    baseline = table.residual_spectrum_many_counts(
        counts,
        method="cython_log_pruned",
        resolving_power=100_000,
    )
    uintphase = table.residual_spectrum_many_counts(
        counts,
        method="cython_log_pruned_uintphase",
        resolving_power=100_000,
    )

    assert _relative_l2(uintphase, baseline) < 1e-10


@pytest.mark.skipif(not has_cython_backend(), reason="Cython backend is not built")
def test_cython_log_pruned_uintphase_threshold_matches_cython_log_pruned():
    table = CenteredLogPhaseTable.build_for_formulas(
        ["C10000H16000N2600O3600S200"],
        elements=("C", "H", "N", "O", "S"),
        dm=0.002,
        min_fft_len=32768,
        resolving_power=100_000,
    )
    counts = table.counts_from_formulas(["C10000H16000N2600O3600S200"])

    baseline = table.residual_spectrum_many_counts(
        counts,
        method="cython_log_pruned",
        resolving_power=100_000,
    )
    threshold = table.residual_spectrum_many_counts(
        counts,
        method="cython_log_pruned_uintphase_threshold",
        resolving_power=100_000,
    )

    assert _relative_l2(threshold, baseline) < 1e-10


@pytest.mark.skipif(not has_cython_backend(), reason="Cython backend is not built")
def test_cython_auto_selects_baseline_for_high_active_fraction():
    table = CenteredLogPhaseTable.build_for_formulas(
        ["C500H800N125O200S10"],
        elements=("C", "H", "N", "O", "S"),
        dm=0.002,
        min_fft_len=32768,
    )
    counts = table.counts_from_formulas(["C500H800N125O200S10"])

    selected = table.select_spectrum_method(counts, method="cython_auto")
    spectrum = table.residual_spectrum_many_counts(counts, method="cython_auto")
    baseline = table.residual_spectrum_many_counts(counts, method="cython_log_pruned")

    assert selected == "cython_log_pruned"
    assert _relative_l2(spectrum, baseline) < 1e-12


@pytest.mark.skipif(not has_cython_backend(), reason="Cython backend is not built")
def test_cython_auto_keeps_baseline_for_gaussian_cutoff():
    table = CenteredLogPhaseTable.build_for_formulas(
        ["C10000H16000N2600O3600S200"],
        elements=("C", "H", "N", "O", "S"),
        dm=0.002,
        min_fft_len=32768,
        resolving_power=100_000,
    )
    counts = table.counts_from_formulas(["C10000H16000N2600O3600S200"])

    selected = table.select_spectrum_method(
        counts,
        method="cython_auto",
        resolving_power=100_000,
    )
    _, _, info = table.mass_profile_window_many_counts(
        counts,
        residual_start=-0.1,
        residual_stop=0.1,
        output_dm=0.01,
        method="cython_auto",
        resolving_power=100_000,
    )

    assert selected == "cython_log_pruned"
    assert info["method"] == "cython_log_pruned"
    assert info["requested_method"] == "cython_auto"


@pytest.mark.skipif(not has_cython_backend(), reason="Cython backend is not built")
def test_cython_auto_no_broadening_uses_template_counts():
    table = CenteredLogPhaseTable.build_for_formulas(
        [
            "C5000H8000N1300O1800S100",
            "C10000H16000N2600O3600S200",
        ],
        elements=("C", "H", "N", "O", "S"),
        dm=0.002,
        min_fft_len=32768,
    )
    lower_counts = table.counts_from_formulas(["C5000H8000N1300O1800S100"])
    higher_counts = table.counts_from_formulas(["C10000H16000N2600O3600S200"])
    carbon_idx = table.elements.index("C")

    assert lower_counts[0, carbon_idx] < table.attenuation_template_trigger_count[carbon_idx]
    assert higher_counts[0, carbon_idx] >= table.attenuation_template_trigger_count[carbon_idx]
    assert table.select_spectrum_method(lower_counts, method="cython_auto") == "cython_log_pruned"
    assert (
        table.select_spectrum_method(higher_counts, method="cython_auto")
        == "cython_log_pruned_uintphase_threshold"
    )


@pytest.mark.skipif(not has_cython_backend(), reason="Cython backend is not built")
def test_float32_attenuation_methods_match_float64_baseline():
    formula = "C500H800N125O200S10"
    baseline_table = CenteredLogPhaseTable.build_for_formulas(
        [formula],
        elements=("C", "H", "N", "O", "S"),
        dm=0.002,
        min_fft_len=32768,
        resolving_power=100_000,
    )
    float32_table = CenteredLogPhaseTable.build_for_formulas(
        [formula],
        elements=("C", "H", "N", "O", "S"),
        dm=0.002,
        min_fft_len=32768,
        resolving_power=100_000,
        attenuation_dtype="float32",
    )
    counts = baseline_table.counts_from_formulas([formula])

    baseline = baseline_table.residual_spectrum_many_counts(
        counts,
        method="cython_log_pruned",
        resolving_power=100_000,
    )
    attn32 = float32_table.residual_spectrum_many_counts(
        counts,
        method="cython_log_pruned_attn32",
        resolving_power=100_000,
    )
    attn32_uint = float32_table.residual_spectrum_many_counts(
        counts,
        method="cython_log_pruned_attn32_uintphase",
        resolving_power=100_000,
    )

    assert float32_table.attenuation.dtype == np.float32
    assert float32_table.has_phase_table
    assert _relative_l2(attn32, baseline) < 1e-7
    assert _relative_l2(attn32_uint, baseline) < 1e-7


@pytest.mark.skipif(not has_cython_backend(), reason="Cython backend is not built")
def test_cython_parallel_workers_match_serial():
    formulas = [
        f"C{500 + i * 3}H{800 + i * 5}N{125 + i}O{200 + i}S{10 + i}"
        for i in range(8)
    ]
    table = CenteredLogPhaseTable.build_for_formulas(
        formulas,
        elements=("C", "H", "N", "O", "S"),
        dm=0.002,
        min_fft_len=32768,
        resolving_power=100_000,
        storage_mode="production",
    )
    counts = table.counts_from_formulas(formulas)

    serial = table.residual_spectrum_many_counts(
        counts,
        method="cython_auto",
        resolving_power=100_000,
        workers=1,
    )
    parallel = table.residual_spectrum_many_counts(
        counts,
        method="cython_auto",
        resolving_power=100_000,
        workers=4,
    )
    _, _, info = table.mass_profile_window_many_counts(
        counts,
        residual_start=-0.1,
        residual_stop=0.1,
        output_dm=0.01,
        method="cython_auto",
        resolving_power=100_000,
        workers=4,
    )

    np.testing.assert_allclose(parallel, serial, rtol=0.0, atol=0.0)
    assert info["workers"] == 4


@pytest.mark.skipif(not has_cython_backend(), reason="Cython backend is not built")
def test_production_storage_discards_cyclephase_and_auto_uses_float32_phase():
    formula = "C500H800N125O200S10"
    research_table = CenteredLogPhaseTable.build_for_formulas(
        [formula],
        elements=("C", "H", "N", "O", "S"),
        dm=0.002,
        min_fft_len=32768,
        resolving_power=100_000,
    )
    production_table = CenteredLogPhaseTable.build_for_formulas(
        [formula],
        elements=("C", "H", "N", "O", "S"),
        dm=0.002,
        min_fft_len=32768,
        resolving_power=100_000,
        storage_mode="production",
    )
    counts = research_table.counts_from_formulas([formula])

    baseline = research_table.residual_spectrum_many_counts(
        counts,
        method="cython_log_pruned",
        resolving_power=100_000,
    )
    production = production_table.residual_spectrum_many_counts(
        counts,
        method="cython_auto",
        resolving_power=100_000,
    )

    assert production_table.storage_mode == "production"
    assert production_table.attenuation.dtype == np.float32
    assert production_table.has_phase_table
    assert production_table.phase_cycles.shape == (0, 0)
    assert production_table.table_nbytes < research_table.table_nbytes
    assert (
        production_table.select_spectrum_method(
            counts,
            method="cython_auto",
            resolving_power=100_000,
        )
        == "cython_log_pruned_attn32"
    )
    assert _relative_l2(production, baseline) < 1e-7


@pytest.mark.skipif(not has_cython_backend(), reason="Cython backend is not built")
def test_minimal_storage_discards_phase_and_auto_uses_uintphase():
    formula = "C500H800N125O200S10"
    table = CenteredLogPhaseTable.build_for_formulas(
        [formula],
        elements=("C", "H", "N", "O", "S"),
        dm=0.002,
        min_fft_len=32768,
        resolving_power=100_000,
        storage_mode="minimal",
    )
    counts = table.counts_from_formulas([formula])

    assert table.storage_mode == "minimal"
    assert table.attenuation.dtype == np.float32
    assert not table.has_phase_table
    assert (
        table.select_spectrum_method(
            counts,
            method="cython_auto",
            resolving_power=100_000,
        )
        == "cython_log_pruned_attn32_uintphase"
    )


@pytest.mark.skipif(not has_cython_backend(), reason="Cython backend is not built")
def test_production_auto_no_broadening_uses_float32_threshold():
    formula = "C10000H16000N2600O3600S200"
    table = CenteredLogPhaseTable.build_for_formulas(
        [formula],
        elements=("C", "H", "N", "O", "S"),
        dm=0.002,
        min_fft_len=32768,
        storage_mode="production",
    )
    counts = table.counts_from_formulas([formula])

    assert (
        table.select_spectrum_method(counts, method="cython_auto")
        == "cython_log_pruned_attn32_uintphase_threshold"
    )


def test_mass_profile_sum_is_preserved():
    table = CenteredLogPhaseTable.build(
        elements=("C", "H", "N", "O", "S"),
        dm=0.01,
        min_fft_len=2048,
    )
    counts = table.counts_from_formulas(["C100H160N25O40S2"])

    mass_axis, profiles, info = table.mass_profile_many_counts(counts)

    assert mass_axis.shape == profiles.shape
    assert info["n_fft"] == table.n_fft
    np.testing.assert_allclose(profiles.sum(axis=-1), np.array([1.0]), atol=1e-12)


def test_profile_sigma_grows_with_formula_size():
    table = CenteredLogPhaseTable.build(
        elements=("C", "H", "N", "O", "S"),
        dm=0.01,
        min_fft_len=2048,
    )
    counts = table.counts_from_formulas(["C10H16O2", "C500H800N125O200S10"])

    sigma = table.profile_sigma_many_counts(counts, resolving_power=100_000)

    assert sigma[1] > sigma[0]


def test_suggest_n_fft_for_counts_expands_large_molecule_window():
    table = CenteredLogPhaseTable.build(
        elements=("C", "H", "N", "O", "S"),
        dm=0.002,
        min_fft_len=32768,
    )
    counts = table.counts_from_formulas(["C10000H16000N2600O3600S200"])

    n_fft = table.suggest_n_fft_for_counts(
        counts,
        min_fft_len=32768,
        safety_sigma=6.0,
        resolving_power=100_000,
    )

    assert n_fft > table.n_fft
    assert n_fft * table.dm > 2 * 6 * table.profile_sigma_many_counts(
        counts,
        resolving_power=100_000,
    )[0]


def test_build_for_formulas_uses_sized_fft_window():
    table = CenteredLogPhaseTable.build_for_formulas(
        ["C10000H16000N2600O3600S200"],
        elements=("C", "H", "N", "O", "S"),
        dm=0.002,
        min_fft_len=32768,
        safety_sigma=6.0,
        resolving_power=100_000,
    )

    assert table.n_fft > 32805


def test_mass_profile_monoisotopic_peak_is_on_low_mass_side():
    table = CenteredLogPhaseTable.build(
        elements=("C", "H", "N", "O", "S"),
        dm=0.002,
        min_fft_len=32768,
    )
    counts = table.counts_from_formulas(["C6H12O6"])

    mass_axis, profiles, _ = table.mass_profile_many_counts(counts)
    peak_mass = float(mass_axis[0, np.argmax(profiles[0])])

    assert abs(peak_mass - 180.06338810219998) < 0.002


def test_czt_window_profile_matches_full_profile_on_table_grid():
    table = CenteredLogPhaseTable.build(
        elements=("C", "H", "N", "O", "S"),
        dm=0.002,
        min_fft_len=32768,
    )
    counts = table.counts_from_formulas(["C100H160N25O40S2"])

    mass_axis, full_profile, _ = table.mass_profile_many_counts(
        counts,
        method="cython_log_pruned" if has_cython_backend() else "log_pruned",
        resolving_power=100_000,
    )
    window_mass, window_profile, info = table.mass_profile_window_many_counts(
        counts,
        residual_start=-0.2,
        residual_stop=0.2,
        output_dm=table.dm,
        method="cython_log_pruned" if has_cython_backend() else "log_pruned",
        resolving_power=100_000,
    )

    indices = np.searchsorted(mass_axis[0], window_mass[0])
    np.testing.assert_allclose(window_mass[0], mass_axis[0, indices], atol=1e-10)
    np.testing.assert_allclose(
        window_profile[0],
        full_profile[0, indices],
        rtol=1e-8,
        atol=1e-12,
    )
    assert info["transform"] == "czt_irfft_window"
    assert info["n_points"] == window_profile.shape[-1]


def test_czt_window_profile_supports_finer_output_spacing():
    table = CenteredLogPhaseTable.build(
        elements=("C", "H", "N", "O", "S"),
        dm=0.004,
        min_fft_len=8192,
    )
    counts = table.counts_from_formulas(["C100H160N25O40S2"])

    mass_axis, profile, info = table.mass_profile_window_many_counts(
        counts,
        residual_start=-0.05,
        residual_stop=0.05,
        output_dm=0.001,
        method="log_pruned",
        resolving_power=100_000,
    )

    assert mass_axis.shape == profile.shape
    assert profile.shape[-1] > 90
    assert info["output_dm"] == pytest.approx(0.001)
    assert np.all(np.isfinite(profile))
    assert np.max(profile) > 0.0


def test_czt_window_profile_supports_absolute_mass_window():
    table = CenteredLogPhaseTable.build(
        elements=("C", "H", "N", "O", "S"),
        dm=0.002,
        min_fft_len=32768,
    )
    counts = table.counts_from_formulas(["C100H160N25O40S2"])
    mean_mass = table.mean_mass_many_counts(counts)[0]

    residual_mass, residual_profile, _ = table.mass_profile_window_many_counts(
        counts,
        residual_start=-0.1,
        residual_stop=0.1,
        output_dm=table.dm,
        method="log_pruned",
        resolving_power=100_000,
    )
    absolute_mass, absolute_profile, _ = table.mass_profile_window_many_counts(
        counts,
        mass_start=mean_mass - 0.1,
        mass_stop=mean_mass + 0.1,
        output_dm=table.dm,
        method="log_pruned",
        resolving_power=100_000,
    )

    np.testing.assert_allclose(absolute_mass, residual_mass, atol=1e-10)
    np.testing.assert_allclose(absolute_profile, residual_profile, rtol=1e-8, atol=1e-14)


def test_large_formula_has_smaller_active_frequency_fraction():
    table = CenteredLogPhaseTable.build(
        elements=("C", "H", "N", "O", "S"),
        dm=0.01,
        min_fft_len=2048,
    )
    counts = table.counts_from_formulas(["C10H16O2", "C500H800N125O200S10"])

    active_fraction = table.active_frequency_fraction(
        counts,
        resolving_power=100_000,
    )

    assert active_fraction[1] < active_fraction[0]


def _relative_l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))
