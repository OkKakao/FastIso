"""Portable tkinter GUI for FastIso."""

from __future__ import annotations

import re
import sys
import threading
from collections.abc import Sequence
from typing import Any

import numpy as np

from .cli import _write_profile_result, simulate_profiles
from .formula import parse_formula

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except ImportError as exc:  # pragma: no cover - depends on Python build.
    raise RuntimeError("FastIso GUI requires tkinter to be installed") from exc


_FORMULA_SPLIT_RE = re.compile(r"[\s,;]+")
_DEFAULT_FORMULA = "C500H800N125O200S10"
_DEFAULT_ELEMENTS = "C H N O S"


def main(argv: Sequence[str] | None = None) -> int:
    """Start the FastIso GUI."""

    argv = list(sys.argv[1:] if argv is None else argv)
    if any(arg in {"-h", "--help"} for arg in argv):
        print("usage: fastiso-gui")
        print()
        print("Start the FastIso tkinter GUI.")
        print("The same GUI can be launched with: python -m fastiso.gui")
        return 0
    if argv:
        print(f"fastiso-gui: unexpected argument {argv[0]!r}", file=sys.stderr)
        return 2

    app = FastIsoGui()
    app.run()
    return 0


class FastIsoGui:
    """Small desktop GUI for interactive FastIso profile simulation."""

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("FastIso")
        self.root.geometry("1120x760")
        self.root.minsize(980, 660)

        self.formula_var = tk.StringVar(value=_DEFAULT_FORMULA)
        self.preset_var = tk.StringVar(value="common")
        self.elements_var = tk.StringVar(value=_DEFAULT_ELEMENTS)
        self.mode_var = tk.StringVar(value="auto")
        self.start_var = tk.StringVar(value="-0.5")
        self.stop_var = tk.StringVar(value="0.5")
        self.output_dm_var = tk.StringVar(value="")
        self.points_var = tk.StringVar(value="")
        self.dm_var = tk.StringVar(value="0.002")
        self.auto_grid_var = tk.BooleanVar(value=True)
        self.samples_per_fwhm_var = tk.StringVar(value="8.0")
        self.min_fft_var = tk.StringVar(value="auto")
        self.safety_sigma_var = tk.StringVar(value="6.0")
        self.rp_var = tk.StringVar(value="100000")
        self.gaussian_sigma_var = tk.StringVar(value="")
        self.method_var = tk.StringVar(value="cython_auto")
        self.storage_var = tk.StringVar(value="auto")
        self.workers_var = tk.StringVar(value="")
        self.auto_window_sigma_var = tk.StringVar(value="6.0")
        self.auto_window_min_half_width_var = tk.StringVar(value="0.1")
        self.output_format_var = tk.StringVar(value="csv")
        self.status_var = tk.StringVar(value="Ready")

        self.result: dict[str, Any] | None = None
        self._worker: threading.Thread | None = None

        self._build_ui()
        self._sync_mode_state()
        self._sync_grid_state()

    def run(self) -> None:
        self.root.mainloop()

    def _build_ui(self) -> None:
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        controls = ttk.Frame(self.root, padding=12)
        controls.grid(row=0, column=0, sticky="ns")
        controls.columnconfigure(1, weight=1)

        output = ttk.Frame(self.root, padding=(0, 12, 12, 12))
        output.grid(row=0, column=1, sticky="nsew")
        output.columnconfigure(0, weight=1)
        output.rowconfigure(3, weight=1)

        row = 0
        row = self._add_entry(controls, row, "Formula", self.formula_var, width=36)
        row = self._add_entry(controls, row, "Preset", self.preset_var, width=36)
        row = self._add_entry(controls, row, "Elements", self.elements_var, width=36)

        ttk.Label(controls, text="Mode").grid(row=row, column=0, sticky="w", pady=4)
        mode = ttk.Combobox(
            controls,
            textvariable=self.mode_var,
            values=("auto", "full"),
            state="readonly",
            width=34,
        )
        mode.grid(row=row, column=1, sticky="ew", pady=4)
        mode.bind("<<ComboboxSelected>>", lambda _event: self._sync_mode_state())
        row += 1

        self.start_entry = self._entry(controls, row, "Start", self.start_var)
        row += 1
        self.stop_entry = self._entry(controls, row, "Stop", self.stop_var)
        row += 1
        self.output_dm_entry = self._entry(controls, row, "Output dm", self.output_dm_var)
        row += 1
        self.points_entry = self._entry(controls, row, "Points", self.points_var)
        row += 1

        ttk.Separator(controls).grid(row=row, column=0, columnspan=2, sticky="ew", pady=10)
        row += 1

        self.dm_entry = self._entry(controls, row, "Table dm", self.dm_var)
        row += 1
        ttk.Checkbutton(
            controls,
            text="Auto grid",
            variable=self.auto_grid_var,
            command=self._sync_grid_state,
        ).grid(row=row, column=0, columnspan=2, sticky="w", pady=4)
        row += 1
        row = self._add_entry(
            controls,
            row,
            "Samples/FWHM",
            self.samples_per_fwhm_var,
            width=36,
        )
        row = self._add_entry(controls, row, "Min FFT", self.min_fft_var, width=36)
        row = self._add_entry(controls, row, "Safety sigma", self.safety_sigma_var, width=36)
        row = self._add_entry(controls, row, "Resolving power", self.rp_var, width=36)
        row = self._add_entry(
            controls,
            row,
            "Gaussian sigma",
            self.gaussian_sigma_var,
            width=36,
        )

        ttk.Label(controls, text="Method").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Combobox(
            controls,
            textvariable=self.method_var,
            values=(
                "cython_auto",
                "cython_log_pruned",
                "cython_log_pruned_attn32_uintphase_threshold",
                "log_pruned",
                "log_table",
                "direct_rebuild",
            ),
            width=34,
        ).grid(row=row, column=1, sticky="ew", pady=4)
        row += 1

        ttk.Label(controls, text="Storage").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Combobox(
            controls,
            textvariable=self.storage_var,
            values=("auto", "research", "production", "minimal"),
            state="readonly",
            width=34,
        ).grid(row=row, column=1, sticky="ew", pady=4)
        row += 1

        row = self._add_entry(controls, row, "Workers", self.workers_var, width=36)
        row = self._add_entry(
            controls,
            row,
            "Auto window sigma",
            self.auto_window_sigma_var,
            width=36,
        )
        row = self._add_entry(
            controls,
            row,
            "Auto min half-width",
            self.auto_window_min_half_width_var,
            width=36,
        )

        ttk.Separator(controls).grid(row=row, column=0, columnspan=2, sticky="ew", pady=10)
        row += 1

        self.parse_button = ttk.Button(controls, text="Parse", command=self.parse_preview)
        self.parse_button.grid(row=row, column=0, sticky="ew", pady=4)
        self.run_button = ttk.Button(controls, text="Run", command=self.run_simulation)
        self.run_button.grid(row=row, column=1, sticky="ew", pady=4)
        row += 1

        ttk.Label(controls, text="Save format").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Combobox(
            controls,
            textvariable=self.output_format_var,
            values=("csv", "json"),
            state="readonly",
            width=34,
        ).grid(row=row, column=1, sticky="ew", pady=4)
        row += 1

        self.save_button = ttk.Button(controls, text="Save", command=self.save_result)
        self.save_button.grid(row=row, column=0, columnspan=2, sticky="ew", pady=4)
        self.save_button.configure(state="disabled")
        row += 1

        ttk.Label(controls, textvariable=self.status_var, wraplength=320).grid(
            row=row,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=(10, 0),
        )

        ttk.Label(output, text="Parsed Formula").grid(row=0, column=0, sticky="w")
        self.parse_text = tk.Text(output, height=5, wrap="none")
        self.parse_text.grid(row=1, column=0, sticky="ew", pady=(4, 10))

        ttk.Label(output, text="Profile Preview").grid(row=2, column=0, sticky="w")
        self.preview = ttk.Treeview(
            output,
            columns=("formula", "mass", "intensity"),
            show="headings",
            height=14,
        )
        self.preview.heading("formula", text="Formula")
        self.preview.heading("mass", text="Mass")
        self.preview.heading("intensity", text="Intensity")
        self.preview.column("formula", width=160, anchor="w")
        self.preview.column("mass", width=160, anchor="e")
        self.preview.column("intensity", width=160, anchor="e")
        self.preview.grid(row=3, column=0, sticky="nsew")

        yscroll = ttk.Scrollbar(output, orient="vertical", command=self.preview.yview)
        yscroll.grid(row=3, column=1, sticky="ns")
        self.preview.configure(yscrollcommand=yscroll.set)

        self.plot = tk.Canvas(output, height=230, background="white", highlightthickness=1)
        self.plot.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        self.plot.bind("<Configure>", lambda _event: self._draw_plot())

        self.summary_text = tk.Text(output, height=8, wrap="word")
        self.summary_text.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(10, 0))

    def _add_entry(
        self,
        parent: ttk.Frame,
        row: int,
        label: str,
        variable: tk.StringVar,
        *,
        width: int,
    ) -> int:
        self._entry(parent, row, label, variable, width=width)
        return row + 1

    def _entry(
        self,
        parent: ttk.Frame,
        row: int,
        label: str,
        variable: tk.StringVar,
        *,
        width: int = 36,
    ) -> ttk.Entry:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=4)
        entry = ttk.Entry(parent, textvariable=variable, width=width)
        entry.grid(row=row, column=1, sticky="ew", pady=4)
        return entry

    def _sync_mode_state(self) -> None:
        mode = self.mode_var.get()
        start_stop_state = "disabled"
        output_state = "normal" if mode == "auto" else "disabled"
        self.start_entry.configure(state=start_stop_state)
        self.stop_entry.configure(state=start_stop_state)
        self.output_dm_entry.configure(state=output_state)
        self.points_entry.configure(state=output_state)

    def _sync_grid_state(self) -> None:
        state = "disabled" if self.auto_grid_var.get() else "normal"
        self.dm_entry.configure(state=state)

    def parse_preview(self) -> None:
        try:
            formulas = _parse_formula_list(self.formula_var.get())
            lines = []
            for formula in formulas:
                counts = parse_formula(formula)
                counts_text = " ".join(
                    f"{element}{count}" for element, count in sorted(counts.items())
                )
                lines.append(f"{formula}: {counts_text}")
            self._set_text(self.parse_text, "\n".join(lines))
            self.status_var.set("Formula parsed")
        except Exception as exc:
            self.status_var.set(f"Parse error: {exc}")
            messagebox.showerror("FastIso", str(exc), parent=self.root)

    def run_simulation(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        try:
            settings = self._collect_settings()
            self.parse_preview()
        except Exception as exc:
            messagebox.showerror("FastIso", str(exc), parent=self.root)
            return

        self.result = None
        self.save_button.configure(state="disabled")
        self.run_button.configure(state="disabled")
        self.status_var.set("Running simulation...")
        self._clear_preview()

        self._worker = threading.Thread(
            target=self._run_worker,
            args=(settings,),
            daemon=True,
        )
        self._worker.start()

    def _run_worker(self, settings: dict[str, Any]) -> None:
        try:
            result = simulate_profiles(**settings)
        except Exception as exc:
            self.root.after(0, self._finish_error, str(exc))
            return
        self.root.after(0, self._finish_success, result)

    def _finish_error(self, message: str) -> None:
        self.run_button.configure(state="normal")
        self.status_var.set(f"Error: {message}")
        messagebox.showerror("FastIso", message, parent=self.root)

    def _finish_success(self, result: dict[str, Any]) -> None:
        self.result = result
        self.run_button.configure(state="normal")
        self.save_button.configure(state="normal")
        self.status_var.set("Simulation complete")
        self._render_result(result)

    def save_result(self) -> None:
        if self.result is None:
            return
        output_format = self.output_format_var.get()
        suffix = ".json" if output_format == "json" else ".csv"
        path = filedialog.asksaveasfilename(
            parent=self.root,
            defaultextension=suffix,
            filetypes=(
                ("JSON files", "*.json"),
                ("CSV files", "*.csv"),
                ("All files", "*.*"),
            ),
        )
        if not path:
            return
        try:
            _write_profile_result(
                self.result,
                output_format=output_format,
                output_path=path,
            )
        except Exception as exc:
            messagebox.showerror("FastIso", str(exc), parent=self.root)
            return
        self.status_var.set(f"Saved {path}")

    def _collect_settings(self) -> dict[str, Any]:
        formulas = _parse_formula_list(self.formula_var.get())
        elements = _parse_elements(self.elements_var.get())
        mode = self.mode_var.get()
        gaussian_sigma = _optional_float(self.gaussian_sigma_var.get(), "Gaussian sigma")
        resolving_power = _optional_float(self.rp_var.get(), "Resolving power")
        if gaussian_sigma is not None:
            resolving_power = None

        settings: dict[str, Any] = {
            "formulas": formulas,
            "preset": self.preset_var.get().strip() or "common",
            "elements": elements,
            "dm": _required_float(self.dm_var.get(), "Table dm"),
            "auto_grid": bool(self.auto_grid_var.get()),
            "samples_per_fwhm": _required_float(
                self.samples_per_fwhm_var.get(),
                "Samples/FWHM",
            ),
            "min_fft_len": _min_fft_value(self.min_fft_var.get()),
            "safety_sigma": _required_float(self.safety_sigma_var.get(), "Safety sigma"),
            "resolving_power": resolving_power,
            "gaussian_sigma": gaussian_sigma,
            "method": self.method_var.get().strip() or "cython_auto",
            "storage_mode": self.storage_var.get().strip() or "auto",
            "workers": _optional_int(self.workers_var.get(), "Workers"),
        }
        if mode != "full":
            settings["window_mode"] = "auto" if mode == "auto" else mode
            settings["output_dm"] = _optional_float(self.output_dm_var.get(), "Output dm")
            settings["points"] = _optional_int(self.points_var.get(), "Points")
            settings["auto_window_sigma"] = _required_float(
                self.auto_window_sigma_var.get(),
                "Auto window sigma",
            )
            settings["auto_window_min_half_width"] = _required_float(
                self.auto_window_min_half_width_var.get(),
                "Auto min half-width",
            )
        return settings

    def _render_result(self, result: dict[str, Any]) -> None:
        self._render_preview(result)
        self._draw_plot()
        self._render_summary(result)

    def _render_preview(self, result: dict[str, Any]) -> None:
        self._clear_preview()
        formulas = result["formulas"]
        mass_axis = np.asarray(result["mass_axis"])
        intensity = np.asarray(result["intensity"])
        max_rows = 500
        inserted = 0
        for row_idx, formula in enumerate(formulas):
            row_mass = np.asarray(mass_axis[row_idx], dtype=np.float64)
            row_intensity = np.asarray(intensity[row_idx], dtype=np.float64)
            row_indices = _profile_preview_indices(
                row_mass,
                row_intensity,
                max_rows=max(1, (max_rows - inserted) // (len(formulas) - row_idx)),
            )
            for point_idx in row_indices:
                if inserted >= max_rows:
                    return
                self.preview.insert(
                    "",
                    "end",
                    values=(
                        formula,
                        f"{float(row_mass[point_idx]):.8f}",
                        f"{float(row_intensity[point_idx]):.8g}",
                    ),
                )
                inserted += 1

    def _render_summary(self, result: dict[str, Any]) -> None:
        metadata = result["metadata"]
        lines = [
            f"Formulas: {', '.join(result['formulas'])}",
            f"Method: {metadata['method']}",
            f"Transform: {metadata['transform']}",
            f"Profile backend: {metadata.get('profile_backend', 'ft')}",
            f"Resource: {metadata['resource']} ({metadata['isotope_data_version']})",
            f"Spectral elements: {', '.join(metadata['spectral_elements']) or '(none)'}",
            f"dm: {metadata['dm']}",
            f"auto grid: {metadata['auto_grid']}",
            f"min FFT: {metadata.get('requested_min_fft_len', metadata.get('min_fft_len'))}",
            f"n_fft: {metadata['n_fft']}",
            f"points: {metadata['n_points']}",
            f"table memory: {metadata['table_nbytes']} bytes",
        ]
        if "window_mode" in metadata:
            requested = metadata.get("requested_window_mode")
            mode_text = metadata["window_mode"]
            if requested is not None:
                mode_text = f"{requested} -> {mode_text}"
            lines.append(
                "window: "
                f"{mode_text} "
                f"[{metadata['window_start']:.6g}, {metadata['window_stop']:.6g}]"
            )
        if "auto_window_method" in metadata:
            lines.append(f"auto window method: {metadata['auto_window_method']}")
        self._set_text(self.summary_text, "\n".join(lines))

    def _draw_plot(self) -> None:
        self.plot.delete("all")
        width = max(self.plot.winfo_width(), 2)
        height = max(self.plot.winfo_height(), 2)
        pad = 28
        self.plot.create_rectangle(pad, 12, width - 12, height - pad, outline="#b0b0b0")
        if self.result is None:
            return

        mass_axis = np.asarray(self.result["mass_axis"])
        intensity = np.asarray(self.result["intensity"])
        if mass_axis.size == 0 or intensity.size == 0:
            return
        x = np.asarray(mass_axis[0], dtype=np.float64)
        y = np.asarray(intensity[0], dtype=np.float64)
        finite = np.isfinite(x) & np.isfinite(y)
        x = x[finite]
        y = y[finite]
        if x.size < 2:
            return
        x, y = _profile_plot_points(x, y, max_points=1500)
        x_min = float(np.min(x))
        x_max = float(np.max(x))
        y_min = float(np.min(y))
        y_max = float(np.max(y))
        if x_max == x_min:
            x_max = x_min + 1.0
        if y_max == y_min:
            y_max = y_min + 1.0
        plot_w = width - pad - 12
        plot_h = height - pad - 12
        coords: list[float] = []
        for x_val, y_val in zip(x, y):
            px = pad + (float(x_val) - x_min) / (x_max - x_min) * plot_w
            py = 12 + (1.0 - (float(y_val) - y_min) / (y_max - y_min)) * plot_h
            coords.extend((px, py))
        if len(coords) >= 4:
            self.plot.create_line(*coords, fill="#1f6feb", width=2)
        if y_min <= 0.0 <= y_max:
            baseline = 12 + (1.0 - (0.0 - y_min) / (y_max - y_min)) * plot_h
            self.plot.create_line(pad, baseline, width - 12, baseline, fill="#808080")

    def _clear_preview(self) -> None:
        for item in self.preview.get_children():
            self.preview.delete(item)
        self.plot.delete("all")
        self._set_text(self.summary_text, "")

    def _set_text(self, widget: tk.Text, text: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.configure(state="disabled")


def _parse_formula_list(text: str) -> list[str]:
    formulas = [part for part in _FORMULA_SPLIT_RE.split(text.strip()) if part]
    if not formulas:
        raise ValueError("Formula is required")
    for formula in formulas:
        parse_formula(formula)
    return formulas


def _profile_preview_indices(
    mass_axis: np.ndarray,
    intensity: np.ndarray,
    *,
    max_rows: int,
) -> np.ndarray:
    """Return peak-focused row indices for the table preview."""

    max_rows = int(max_rows)
    if max_rows <= 0:
        return np.array([], dtype=np.int64)
    finite = np.isfinite(mass_axis) & np.isfinite(intensity)
    finite_indices = np.flatnonzero(finite)
    if finite_indices.size <= max_rows:
        return finite_indices

    y = intensity[finite]
    local = _local_peak_indices(y)
    if local.size:
        peak_indices = finite_indices[local]
        peak_order = np.argsort(intensity[peak_indices])[::-1]
        selected = peak_indices[peak_order[:max_rows]]
        return np.array(sorted(selected), dtype=np.int64)

    order = np.argsort(y)[::-1]
    selected = finite_indices[order[:max_rows]]
    return np.array(sorted(selected), dtype=np.int64)


def _profile_plot_points(
    mass_axis: np.ndarray,
    intensity: np.ndarray,
    *,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Downsample a dense profile while preserving narrow local peaks."""

    max_points = int(max_points)
    if mass_axis.size <= max_points or max_points < 4:
        return mass_axis, intensity

    peak_budget = max(1, max_points // 4)
    bucket_count = max(1, (max_points - peak_budget - 4) // 2)
    edges = np.linspace(0, mass_axis.size, bucket_count + 1, dtype=np.int64)
    indices: set[int] = {0, int(mass_axis.size - 1)}
    for start, stop in zip(edges[:-1], edges[1:]):
        if stop <= start:
            continue
        segment = intensity[start:stop]
        indices.add(int(start + np.argmin(segment)))
        indices.add(int(start + np.argmax(segment)))

    local = _local_peak_indices(intensity)
    if local.size:
        peak_order = np.argsort(intensity[local])[::-1]
        indices.update(int(index) for index in local[peak_order[:peak_budget]])

    ordered = np.array(sorted(indices), dtype=np.int64)
    if ordered.size > max_points:
        keep = np.linspace(0, ordered.size - 1, max_points, dtype=np.int64)
        ordered = ordered[keep]
    return mass_axis[ordered], intensity[ordered]


def _local_peak_indices(intensity: np.ndarray) -> np.ndarray:
    if intensity.size == 0:
        return np.array([], dtype=np.int64)
    if intensity.size == 1:
        if intensity[0] > 0.0:
            return np.array([0], dtype=np.int64)
        return np.array([], dtype=np.int64)
    peaks = []
    if intensity[0] > intensity[1] and intensity[0] > 0.0:
        peaks.append(0)
    middle = np.flatnonzero(
        (intensity[1:-1] >= intensity[:-2])
        & (intensity[1:-1] >= intensity[2:])
        & (intensity[1:-1] > 0.0)
    ) + 1
    peaks.extend(int(index) for index in middle)
    if intensity[-1] > intensity[-2] and intensity[-1] > 0.0:
        peaks.append(int(intensity.size - 1))
    return np.array(peaks, dtype=np.int64)


def _parse_elements(text: str) -> list[str] | None:
    elements = [part for part in _FORMULA_SPLIT_RE.split(text.strip()) if part]
    return elements or None


def _required_float(text: str, label: str) -> float:
    value = _optional_float(text, label)
    if value is None:
        raise ValueError(f"{label} is required")
    return value


def _optional_float(text: str, label: str) -> float | None:
    stripped = text.strip()
    if not stripped:
        return None
    try:
        return float(stripped)
    except ValueError as exc:
        raise ValueError(f"{label} must be a number") from exc


def _required_int(text: str, label: str) -> int:
    value = _optional_int(text, label)
    if value is None:
        raise ValueError(f"{label} is required")
    return value


def _min_fft_value(text: str) -> str | int:
    stripped = text.strip()
    if not stripped or stripped.lower() == "auto":
        return "auto"
    return _required_int(stripped, "Min FFT")


def _optional_int(text: str, label: str) -> int | None:
    stripped = text.strip()
    if not stripped:
        return None
    try:
        value = int(stripped)
    except ValueError as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if value < 1:
        raise ValueError(f"{label} must be positive")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
