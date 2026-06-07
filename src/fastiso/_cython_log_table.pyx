# cython: boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True

from libc.math cimport cos, exp, fmod, log, sin, sqrt
from cython.parallel cimport prange

import numpy as np


cdef double TWO_PI = 6.2831853071795864769252867665590057683943387987502
cdef double INV_TWO64 = 5.42101086242752217003726400434970855712890625e-20


cdef Py_ssize_t gaussian_k_limit(
    double[::1] omega,
    double sigma,
    double max_attenuation,
) noexcept nogil:
    cdef Py_ssize_t n_freq = omega.shape[0]
    cdef Py_ssize_t lo, hi, mid
    cdef double omega_limit

    if sigma == 0.0:
        return n_freq
    omega_limit = sqrt(2.0 * max_attenuation) / sigma
    lo = 0
    hi = n_freq
    while lo < hi:
        mid = (lo + hi) // 2
        if omega[mid] <= omega_limit:
            lo = mid + 1
        else:
            hi = mid
    return lo


cdef void fill_log_pruned_row(
    Py_ssize_t i,
    long long[:, ::1] counts,
    double[:, ::1] attenuation_table,
    double[:, ::1] phase_table,
    double[::1] omega,
    double[::1] gaussian_sigma,
    Py_ssize_t n_elements,
    Py_ssize_t n_sigma,
    double max_attenuation,
    double complex[:, ::1] out_view,
) noexcept nogil:
    cdef Py_ssize_t e, k, k_limit
    cdef long long count
    cdef double attenuation, phase, magnitude, sigma, omega_k

    sigma = 0.0
    if n_sigma != 0:
        sigma = gaussian_sigma[i]
    k_limit = gaussian_k_limit(omega, sigma, max_attenuation)
    for k in range(k_limit):
        attenuation = 0.0
        for e in range(n_elements):
            count = counts[i, e]
            if count != 0:
                attenuation += count * attenuation_table[e, k]
        if sigma != 0.0:
            omega_k = omega[k]
            attenuation += 0.5 * sigma * sigma * omega_k * omega_k
        if attenuation <= max_attenuation:
            phase = 0.0
            for e in range(n_elements):
                count = counts[i, e]
                if count != 0:
                    phase += count * phase_table[e, k]
            magnitude = exp(-attenuation)
            out_view[i, k] = magnitude * (cos(phase) + 1j * sin(phase))


cdef void fill_log_pruned_uintphase_row(
    Py_ssize_t i,
    long long[:, ::1] counts,
    double[:, ::1] attenuation_table,
    unsigned long long[:, ::1] phase_uint64_table,
    double[::1] omega,
    double[::1] gaussian_sigma,
    Py_ssize_t n_elements,
    Py_ssize_t n_sigma,
    double max_attenuation,
    double complex[:, ::1] out_view,
) noexcept nogil:
    cdef Py_ssize_t e, k, k_limit
    cdef long long count
    cdef unsigned long long phase_acc
    cdef double attenuation, phase, magnitude, sigma, omega_k

    sigma = 0.0
    if n_sigma != 0:
        sigma = gaussian_sigma[i]
    k_limit = gaussian_k_limit(omega, sigma, max_attenuation)
    for k in range(k_limit):
        attenuation = 0.0
        for e in range(n_elements):
            count = counts[i, e]
            if count != 0:
                attenuation += count * attenuation_table[e, k]
        if sigma != 0.0:
            omega_k = omega[k]
            attenuation += 0.5 * sigma * sigma * omega_k * omega_k
        if attenuation <= max_attenuation:
            phase_acc = 0
            for e in range(n_elements):
                count = counts[i, e]
                if count != 0:
                    phase_acc += (<unsigned long long> count) * phase_uint64_table[e, k]
            phase = TWO_PI * (<double> phase_acc) * INV_TWO64
            magnitude = exp(-attenuation)
            out_view[i, k] = magnitude * (cos(phase) + 1j * sin(phase))


cdef void fill_log_pruned_uintphase_threshold_row(
    Py_ssize_t i,
    long long[:, ::1] counts,
    double[:, ::1] attenuation_table,
    unsigned long long[:, ::1] phase_uint64_table,
    unsigned int[:, ::1] count_threshold_table,
    double[::1] omega,
    double[::1] gaussian_sigma,
    Py_ssize_t n_elements,
    Py_ssize_t n_sigma,
    double max_attenuation,
    double complex[:, ::1] out_view,
) noexcept nogil:
    cdef Py_ssize_t e, k, k_limit
    cdef long long count
    cdef unsigned long long ucount, phase_acc
    cdef unsigned int threshold_count
    cdef bint threshold_pruned
    cdef double attenuation, phase, magnitude, sigma, omega_k

    sigma = 0.0
    if n_sigma != 0:
        sigma = gaussian_sigma[i]
    k_limit = gaussian_k_limit(omega, sigma, max_attenuation)
    for k in range(k_limit):
        attenuation = 0.0
        if sigma != 0.0:
            omega_k = omega[k]
            attenuation = 0.5 * sigma * sigma * omega_k * omega_k
            if attenuation > max_attenuation:
                continue

        threshold_pruned = False
        for e in range(n_elements):
            count = counts[i, e]
            if count != 0:
                ucount = <unsigned long long> count
                threshold_count = count_threshold_table[e, k]
                if threshold_count != 0 and ucount >= threshold_count:
                    threshold_pruned = True
                    break
        if threshold_pruned:
            continue

        for e in range(n_elements):
            count = counts[i, e]
            if count != 0:
                attenuation += count * attenuation_table[e, k]
        if attenuation <= max_attenuation:
            phase_acc = 0
            for e in range(n_elements):
                count = counts[i, e]
                if count != 0:
                    phase_acc += (<unsigned long long> count) * phase_uint64_table[e, k]
            phase = TWO_PI * (<double> phase_acc) * INV_TWO64
            magnitude = exp(-attenuation)
            out_view[i, k] = magnitude * (cos(phase) + 1j * sin(phase))


cdef void fill_log_pruned_attenuation32_row(
    Py_ssize_t i,
    long long[:, ::1] counts,
    float[:, ::1] attenuation_table,
    double[:, ::1] phase_table,
    double[::1] omega,
    double[::1] gaussian_sigma,
    Py_ssize_t n_elements,
    Py_ssize_t n_sigma,
    double max_attenuation,
    double complex[:, ::1] out_view,
) noexcept nogil:
    cdef Py_ssize_t e, k, k_limit
    cdef long long count
    cdef double attenuation, phase, magnitude, sigma, omega_k

    sigma = 0.0
    if n_sigma != 0:
        sigma = gaussian_sigma[i]
    k_limit = gaussian_k_limit(omega, sigma, max_attenuation)
    for k in range(k_limit):
        attenuation = 0.0
        for e in range(n_elements):
            count = counts[i, e]
            if count != 0:
                attenuation += count * attenuation_table[e, k]
        if sigma != 0.0:
            omega_k = omega[k]
            attenuation += 0.5 * sigma * sigma * omega_k * omega_k
        if attenuation <= max_attenuation:
            phase = 0.0
            for e in range(n_elements):
                count = counts[i, e]
                if count != 0:
                    phase += count * phase_table[e, k]
            magnitude = exp(-attenuation)
            out_view[i, k] = magnitude * (cos(phase) + 1j * sin(phase))


cdef void fill_log_pruned_attenuation32_uintphase_row(
    Py_ssize_t i,
    long long[:, ::1] counts,
    float[:, ::1] attenuation_table,
    unsigned long long[:, ::1] phase_uint64_table,
    double[::1] omega,
    double[::1] gaussian_sigma,
    Py_ssize_t n_elements,
    Py_ssize_t n_sigma,
    double max_attenuation,
    double complex[:, ::1] out_view,
) noexcept nogil:
    cdef Py_ssize_t e, k, k_limit
    cdef long long count
    cdef unsigned long long phase_acc
    cdef double attenuation, phase, magnitude, sigma, omega_k

    sigma = 0.0
    if n_sigma != 0:
        sigma = gaussian_sigma[i]
    k_limit = gaussian_k_limit(omega, sigma, max_attenuation)
    for k in range(k_limit):
        attenuation = 0.0
        for e in range(n_elements):
            count = counts[i, e]
            if count != 0:
                attenuation += count * attenuation_table[e, k]
        if sigma != 0.0:
            omega_k = omega[k]
            attenuation += 0.5 * sigma * sigma * omega_k * omega_k
        if attenuation <= max_attenuation:
            phase_acc = 0
            for e in range(n_elements):
                count = counts[i, e]
                if count != 0:
                    phase_acc += (<unsigned long long> count) * phase_uint64_table[e, k]
            phase = TWO_PI * (<double> phase_acc) * INV_TWO64
            magnitude = exp(-attenuation)
            out_view[i, k] = magnitude * (cos(phase) + 1j * sin(phase))


cdef void fill_log_pruned_attenuation32_uintphase_threshold_row(
    Py_ssize_t i,
    long long[:, ::1] counts,
    float[:, ::1] attenuation_table,
    unsigned long long[:, ::1] phase_uint64_table,
    unsigned int[:, ::1] count_threshold_table,
    double[::1] omega,
    double[::1] gaussian_sigma,
    Py_ssize_t n_elements,
    Py_ssize_t n_sigma,
    double max_attenuation,
    double complex[:, ::1] out_view,
) noexcept nogil:
    cdef Py_ssize_t e, k, k_limit
    cdef long long count
    cdef unsigned long long ucount, phase_acc
    cdef unsigned int threshold_count
    cdef bint threshold_pruned
    cdef double attenuation, phase, magnitude, sigma, omega_k

    sigma = 0.0
    if n_sigma != 0:
        sigma = gaussian_sigma[i]
    k_limit = gaussian_k_limit(omega, sigma, max_attenuation)
    for k in range(k_limit):
        attenuation = 0.0
        if sigma != 0.0:
            omega_k = omega[k]
            attenuation = 0.5 * sigma * sigma * omega_k * omega_k
            if attenuation > max_attenuation:
                continue

        threshold_pruned = False
        for e in range(n_elements):
            count = counts[i, e]
            if count != 0:
                ucount = <unsigned long long> count
                threshold_count = count_threshold_table[e, k]
                if threshold_count != 0 and ucount >= threshold_count:
                    threshold_pruned = True
                    break
        if threshold_pruned:
            continue

        for e in range(n_elements):
            count = counts[i, e]
            if count != 0:
                attenuation += count * attenuation_table[e, k]
        if attenuation <= max_attenuation:
            phase_acc = 0
            for e in range(n_elements):
                count = counts[i, e]
                if count != 0:
                    phase_acc += (<unsigned long long> count) * phase_uint64_table[e, k]
            phase = TWO_PI * (<double> phase_acc) * INV_TWO64
            magnitude = exp(-attenuation)
            out_view[i, k] = magnitude * (cos(phase) + 1j * sin(phase))


def log_table_spectrum(
    long long[:, ::1] counts,
    double[:, ::1] attenuation_table,
    double[:, ::1] phase_table,
):
    """Evaluate full centered spectra with C loops."""

    cdef Py_ssize_t n_formulas = counts.shape[0]
    cdef Py_ssize_t n_elements = counts.shape[1]
    cdef Py_ssize_t n_freq = attenuation_table.shape[1]
    cdef Py_ssize_t i, e, k
    cdef long long count
    cdef double attenuation, phase, magnitude
    cdef double complex[:, ::1] out_view

    if phase_table.shape[0] != n_elements or attenuation_table.shape[0] != n_elements:
        raise ValueError("table element dimension does not match counts")
    if phase_table.shape[1] != n_freq:
        raise ValueError("attenuation and phase table frequency dimensions differ")

    out = np.empty((n_formulas, n_freq), dtype=np.complex128)
    out_view = out
    for i in range(n_formulas):
        for k in range(n_freq):
            attenuation = 0.0
            phase = 0.0
            for e in range(n_elements):
                count = counts[i, e]
                if count != 0:
                    attenuation += count * attenuation_table[e, k]
                    phase += count * phase_table[e, k]
            magnitude = exp(-attenuation)
            out_view[i, k] = magnitude * (cos(phase) + 1j * sin(phase))
    return out


def log_pruned_spectrum(
    long long[:, ::1] counts,
    double[:, ::1] attenuation_table,
    double[:, ::1] phase_table,
    double[::1] omega,
    double[::1] gaussian_sigma,
    double prune_cutoff,
    int workers=1,
):
    """Evaluate centered spectra while skipping inactive frequency bins."""

    cdef Py_ssize_t n_formulas = counts.shape[0]
    cdef Py_ssize_t n_elements = counts.shape[1]
    cdef Py_ssize_t n_freq = attenuation_table.shape[1]
    cdef Py_ssize_t n_sigma = gaussian_sigma.shape[0]
    cdef Py_ssize_t i
    cdef double max_attenuation
    cdef double complex[:, ::1] out_view

    if not (0.0 < prune_cutoff < 1.0):
        raise ValueError("prune_cutoff must be between 0 and 1")
    if phase_table.shape[0] != n_elements or attenuation_table.shape[0] != n_elements:
        raise ValueError("table element dimension does not match counts")
    if phase_table.shape[1] != n_freq or omega.shape[0] != n_freq:
        raise ValueError("frequency dimensions differ")
    if n_sigma != 0 and n_sigma != n_formulas:
        raise ValueError("gaussian_sigma must be empty or one value per formula")

    max_attenuation = -log(prune_cutoff)
    out = np.zeros((n_formulas, n_freq), dtype=np.complex128)
    out_view = out
    if workers > 1 and n_formulas > 1:
        for i in prange(n_formulas, nogil=True, num_threads=workers, schedule="static"):
            fill_log_pruned_row(
                i,
                counts,
                attenuation_table,
                phase_table,
                omega,
                gaussian_sigma,
                n_elements,
                n_sigma,
                max_attenuation,
                out_view,
            )
    else:
        for i in range(n_formulas):
            fill_log_pruned_row(
                i,
                counts,
                attenuation_table,
                phase_table,
                omega,
                gaussian_sigma,
                n_elements,
                n_sigma,
                max_attenuation,
                out_view,
            )
    return out


def log_pruned_spectrum_modphase(
    long long[:, ::1] counts,
    double[:, ::1] attenuation_table,
    double[:, ::1] phase_table,
    double[::1] omega,
    double[::1] gaussian_sigma,
    double prune_cutoff,
):
    """Evaluate pruned spectra with phase reduced modulo 2*pi before trig."""

    cdef Py_ssize_t n_formulas = counts.shape[0]
    cdef Py_ssize_t n_elements = counts.shape[1]
    cdef Py_ssize_t n_freq = attenuation_table.shape[1]
    cdef Py_ssize_t n_sigma = gaussian_sigma.shape[0]
    cdef Py_ssize_t i, e, k
    cdef Py_ssize_t k_limit
    cdef long long count
    cdef double max_attenuation
    cdef double attenuation, phase, magnitude, sigma, omega_k
    cdef double complex[:, ::1] out_view

    if not (0.0 < prune_cutoff < 1.0):
        raise ValueError("prune_cutoff must be between 0 and 1")
    if phase_table.shape[0] != n_elements or attenuation_table.shape[0] != n_elements:
        raise ValueError("table element dimension does not match counts")
    if phase_table.shape[1] != n_freq or omega.shape[0] != n_freq:
        raise ValueError("frequency dimensions differ")
    if n_sigma != 0 and n_sigma != n_formulas:
        raise ValueError("gaussian_sigma must be empty or one value per formula")

    max_attenuation = -log(prune_cutoff)
    out = np.zeros((n_formulas, n_freq), dtype=np.complex128)
    out_view = out
    for i in range(n_formulas):
        sigma = 0.0
        if n_sigma != 0:
            sigma = gaussian_sigma[i]
        k_limit = gaussian_k_limit(omega, sigma, max_attenuation)
        for k in range(k_limit):
            attenuation = 0.0
            for e in range(n_elements):
                count = counts[i, e]
                if count != 0:
                    attenuation += count * attenuation_table[e, k]
            if sigma != 0.0:
                omega_k = omega[k]
                attenuation += 0.5 * sigma * sigma * omega_k * omega_k
            if attenuation <= max_attenuation:
                phase = 0.0
                for e in range(n_elements):
                    count = counts[i, e]
                    if count != 0:
                        phase += count * phase_table[e, k]
                phase = fmod(phase, TWO_PI)
                magnitude = exp(-attenuation)
                out_view[i, k] = magnitude * (cos(phase) + 1j * sin(phase))
    return out


def log_pruned_spectrum_cyclephase(
    long long[:, ::1] counts,
    double[:, ::1] attenuation_table,
    double[:, ::1] phase_cycle_table,
    double[::1] omega,
    double[::1] gaussian_sigma,
    double prune_cutoff,
):
    """Evaluate pruned spectra using pre-wrapped phase cycles."""

    cdef Py_ssize_t n_formulas = counts.shape[0]
    cdef Py_ssize_t n_elements = counts.shape[1]
    cdef Py_ssize_t n_freq = attenuation_table.shape[1]
    cdef Py_ssize_t n_sigma = gaussian_sigma.shape[0]
    cdef Py_ssize_t i, e, k
    cdef Py_ssize_t k_limit
    cdef long long count
    cdef long long whole_cycles
    cdef double max_attenuation
    cdef double attenuation, phase_cycles, phase, magnitude, sigma, omega_k
    cdef double complex[:, ::1] out_view

    if not (0.0 < prune_cutoff < 1.0):
        raise ValueError("prune_cutoff must be between 0 and 1")
    if phase_cycle_table.shape[0] != n_elements or attenuation_table.shape[0] != n_elements:
        raise ValueError("table element dimension does not match counts")
    if phase_cycle_table.shape[1] != n_freq or omega.shape[0] != n_freq:
        raise ValueError("frequency dimensions differ")
    if n_sigma != 0 and n_sigma != n_formulas:
        raise ValueError("gaussian_sigma must be empty or one value per formula")

    max_attenuation = -log(prune_cutoff)
    out = np.zeros((n_formulas, n_freq), dtype=np.complex128)
    out_view = out
    for i in range(n_formulas):
        sigma = 0.0
        if n_sigma != 0:
            sigma = gaussian_sigma[i]
        k_limit = gaussian_k_limit(omega, sigma, max_attenuation)
        for k in range(k_limit):
            attenuation = 0.0
            for e in range(n_elements):
                count = counts[i, e]
                if count != 0:
                    attenuation += count * attenuation_table[e, k]
            if sigma != 0.0:
                omega_k = omega[k]
                attenuation += 0.5 * sigma * sigma * omega_k * omega_k
            if attenuation <= max_attenuation:
                phase_cycles = 0.0
                for e in range(n_elements):
                    count = counts[i, e]
                    if count != 0:
                        phase_cycles += count * phase_cycle_table[e, k]
                whole_cycles = <long long> phase_cycles
                phase_cycles = phase_cycles - whole_cycles
                phase = TWO_PI * phase_cycles
                magnitude = exp(-attenuation)
                out_view[i, k] = magnitude * (cos(phase) + 1j * sin(phase))
    return out


def log_pruned_spectrum_uintphase(
    long long[:, ::1] counts,
    double[:, ::1] attenuation_table,
    unsigned long long[:, ::1] phase_uint64_table,
    double[::1] omega,
    double[::1] gaussian_sigma,
    double prune_cutoff,
    int workers=1,
):
    """Evaluate pruned spectra using uint64 phase cycles and overflow modulo."""

    cdef Py_ssize_t n_formulas = counts.shape[0]
    cdef Py_ssize_t n_elements = counts.shape[1]
    cdef Py_ssize_t n_freq = attenuation_table.shape[1]
    cdef Py_ssize_t n_sigma = gaussian_sigma.shape[0]
    cdef Py_ssize_t i
    cdef double max_attenuation
    cdef double complex[:, ::1] out_view

    if not (0.0 < prune_cutoff < 1.0):
        raise ValueError("prune_cutoff must be between 0 and 1")
    if phase_uint64_table.shape[0] != n_elements or attenuation_table.shape[0] != n_elements:
        raise ValueError("table element dimension does not match counts")
    if phase_uint64_table.shape[1] != n_freq or omega.shape[0] != n_freq:
        raise ValueError("frequency dimensions differ")
    if n_sigma != 0 and n_sigma != n_formulas:
        raise ValueError("gaussian_sigma must be empty or one value per formula")

    max_attenuation = -log(prune_cutoff)
    out = np.zeros((n_formulas, n_freq), dtype=np.complex128)
    out_view = out
    if workers > 1 and n_formulas > 1:
        for i in prange(n_formulas, nogil=True, num_threads=workers, schedule="static"):
            fill_log_pruned_uintphase_row(
                i,
                counts,
                attenuation_table,
                phase_uint64_table,
                omega,
                gaussian_sigma,
                n_elements,
                n_sigma,
                max_attenuation,
                out_view,
            )
    else:
        for i in range(n_formulas):
            fill_log_pruned_uintphase_row(
                i,
                counts,
                attenuation_table,
                phase_uint64_table,
                omega,
                gaussian_sigma,
                n_elements,
                n_sigma,
                max_attenuation,
                out_view,
            )
    return out


def log_pruned_spectrum_uintphase_threshold(
    long long[:, ::1] counts,
    double[:, ::1] attenuation_table,
    unsigned long long[:, ::1] phase_uint64_table,
    unsigned int[:, ::1] count_threshold_table,
    double[::1] omega,
    double[::1] gaussian_sigma,
    double prune_cutoff,
    int workers=1,
):
    """Evaluate pruned spectra with per-element count threshold prefilter."""

    cdef Py_ssize_t n_formulas = counts.shape[0]
    cdef Py_ssize_t n_elements = counts.shape[1]
    cdef Py_ssize_t n_freq = attenuation_table.shape[1]
    cdef Py_ssize_t n_sigma = gaussian_sigma.shape[0]
    cdef Py_ssize_t i
    cdef double max_attenuation
    cdef double complex[:, ::1] out_view

    if not (0.0 < prune_cutoff < 1.0):
        raise ValueError("prune_cutoff must be between 0 and 1")
    if phase_uint64_table.shape[0] != n_elements or attenuation_table.shape[0] != n_elements:
        raise ValueError("table element dimension does not match counts")
    if count_threshold_table.shape[0] != n_elements:
        raise ValueError("threshold table element dimension does not match counts")
    if phase_uint64_table.shape[1] != n_freq or omega.shape[0] != n_freq:
        raise ValueError("frequency dimensions differ")
    if count_threshold_table.shape[1] != n_freq:
        raise ValueError("threshold table frequency dimension differs")
    if n_sigma != 0 and n_sigma != n_formulas:
        raise ValueError("gaussian_sigma must be empty or one value per formula")

    max_attenuation = -log(prune_cutoff)
    out = np.zeros((n_formulas, n_freq), dtype=np.complex128)
    out_view = out
    if workers > 1 and n_formulas > 1:
        for i in prange(n_formulas, nogil=True, num_threads=workers, schedule="static"):
            fill_log_pruned_uintphase_threshold_row(
                i,
                counts,
                attenuation_table,
                phase_uint64_table,
                count_threshold_table,
                omega,
                gaussian_sigma,
                n_elements,
                n_sigma,
                max_attenuation,
                out_view,
            )
    else:
        for i in range(n_formulas):
            fill_log_pruned_uintphase_threshold_row(
                i,
                counts,
                attenuation_table,
                phase_uint64_table,
                count_threshold_table,
                omega,
                gaussian_sigma,
                n_elements,
                n_sigma,
                max_attenuation,
                out_view,
            )
    return out


def log_pruned_spectrum_attenuation32(
    long long[:, ::1] counts,
    float[:, ::1] attenuation_table,
    double[:, ::1] phase_table,
    double[::1] omega,
    double[::1] gaussian_sigma,
    double prune_cutoff,
    int workers=1,
):
    """Evaluate pruned spectra with float32 attenuation and double phase."""

    cdef Py_ssize_t n_formulas = counts.shape[0]
    cdef Py_ssize_t n_elements = counts.shape[1]
    cdef Py_ssize_t n_freq = attenuation_table.shape[1]
    cdef Py_ssize_t n_sigma = gaussian_sigma.shape[0]
    cdef Py_ssize_t i
    cdef double max_attenuation
    cdef double complex[:, ::1] out_view

    if not (0.0 < prune_cutoff < 1.0):
        raise ValueError("prune_cutoff must be between 0 and 1")
    if phase_table.shape[0] != n_elements or attenuation_table.shape[0] != n_elements:
        raise ValueError("table element dimension does not match counts")
    if phase_table.shape[1] != n_freq or omega.shape[0] != n_freq:
        raise ValueError("frequency dimensions differ")
    if n_sigma != 0 and n_sigma != n_formulas:
        raise ValueError("gaussian_sigma must be empty or one value per formula")

    max_attenuation = -log(prune_cutoff)
    out = np.zeros((n_formulas, n_freq), dtype=np.complex128)
    out_view = out
    if workers > 1 and n_formulas > 1:
        for i in prange(n_formulas, nogil=True, num_threads=workers, schedule="static"):
            fill_log_pruned_attenuation32_row(
                i,
                counts,
                attenuation_table,
                phase_table,
                omega,
                gaussian_sigma,
                n_elements,
                n_sigma,
                max_attenuation,
                out_view,
            )
    else:
        for i in range(n_formulas):
            fill_log_pruned_attenuation32_row(
                i,
                counts,
                attenuation_table,
                phase_table,
                omega,
                gaussian_sigma,
                n_elements,
                n_sigma,
                max_attenuation,
                out_view,
            )
    return out


def log_pruned_spectrum_attenuation32_uintphase(
    long long[:, ::1] counts,
    float[:, ::1] attenuation_table,
    unsigned long long[:, ::1] phase_uint64_table,
    double[::1] omega,
    double[::1] gaussian_sigma,
    double prune_cutoff,
    int workers=1,
):
    """Evaluate pruned spectra with float32 attenuation and uint64 phase."""

    cdef Py_ssize_t n_formulas = counts.shape[0]
    cdef Py_ssize_t n_elements = counts.shape[1]
    cdef Py_ssize_t n_freq = attenuation_table.shape[1]
    cdef Py_ssize_t n_sigma = gaussian_sigma.shape[0]
    cdef Py_ssize_t i
    cdef double max_attenuation
    cdef double complex[:, ::1] out_view

    if not (0.0 < prune_cutoff < 1.0):
        raise ValueError("prune_cutoff must be between 0 and 1")
    if phase_uint64_table.shape[0] != n_elements or attenuation_table.shape[0] != n_elements:
        raise ValueError("table element dimension does not match counts")
    if phase_uint64_table.shape[1] != n_freq or omega.shape[0] != n_freq:
        raise ValueError("frequency dimensions differ")
    if n_sigma != 0 and n_sigma != n_formulas:
        raise ValueError("gaussian_sigma must be empty or one value per formula")

    max_attenuation = -log(prune_cutoff)
    out = np.zeros((n_formulas, n_freq), dtype=np.complex128)
    out_view = out
    if workers > 1 and n_formulas > 1:
        for i in prange(n_formulas, nogil=True, num_threads=workers, schedule="static"):
            fill_log_pruned_attenuation32_uintphase_row(
                i,
                counts,
                attenuation_table,
                phase_uint64_table,
                omega,
                gaussian_sigma,
                n_elements,
                n_sigma,
                max_attenuation,
                out_view,
            )
    else:
        for i in range(n_formulas):
            fill_log_pruned_attenuation32_uintphase_row(
                i,
                counts,
                attenuation_table,
                phase_uint64_table,
                omega,
                gaussian_sigma,
                n_elements,
                n_sigma,
                max_attenuation,
                out_view,
            )
    return out


def log_pruned_spectrum_attenuation32_uintphase_threshold(
    long long[:, ::1] counts,
    float[:, ::1] attenuation_table,
    unsigned long long[:, ::1] phase_uint64_table,
    unsigned int[:, ::1] count_threshold_table,
    double[::1] omega,
    double[::1] gaussian_sigma,
    double prune_cutoff,
    int workers=1,
):
    """Evaluate pruned spectra with float32 attenuation, uint64 phase, and thresholds."""

    cdef Py_ssize_t n_formulas = counts.shape[0]
    cdef Py_ssize_t n_elements = counts.shape[1]
    cdef Py_ssize_t n_freq = attenuation_table.shape[1]
    cdef Py_ssize_t n_sigma = gaussian_sigma.shape[0]
    cdef Py_ssize_t i
    cdef double max_attenuation
    cdef double complex[:, ::1] out_view

    if not (0.0 < prune_cutoff < 1.0):
        raise ValueError("prune_cutoff must be between 0 and 1")
    if phase_uint64_table.shape[0] != n_elements or attenuation_table.shape[0] != n_elements:
        raise ValueError("table element dimension does not match counts")
    if count_threshold_table.shape[0] != n_elements:
        raise ValueError("threshold table element dimension does not match counts")
    if phase_uint64_table.shape[1] != n_freq or omega.shape[0] != n_freq:
        raise ValueError("frequency dimensions differ")
    if count_threshold_table.shape[1] != n_freq:
        raise ValueError("threshold table frequency dimension differs")
    if n_sigma != 0 and n_sigma != n_formulas:
        raise ValueError("gaussian_sigma must be empty or one value per formula")

    max_attenuation = -log(prune_cutoff)
    out = np.zeros((n_formulas, n_freq), dtype=np.complex128)
    out_view = out
    if workers > 1 and n_formulas > 1:
        for i in prange(n_formulas, nogil=True, num_threads=workers, schedule="static"):
            fill_log_pruned_attenuation32_uintphase_threshold_row(
                i,
                counts,
                attenuation_table,
                phase_uint64_table,
                count_threshold_table,
                omega,
                gaussian_sigma,
                n_elements,
                n_sigma,
                max_attenuation,
                out_view,
            )
    else:
        for i in range(n_formulas):
            fill_log_pruned_attenuation32_uintphase_threshold_row(
                i,
                counts,
                attenuation_table,
                phase_uint64_table,
                count_threshold_table,
                omega,
                gaussian_sigma,
                n_elements,
                n_sigma,
                max_attenuation,
                out_view,
            )
    return out
