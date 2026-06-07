"""Portable tkinter GUI for FastIso."""

from __future__ import annotations

import re
import sys
import threading
from collections.abc import Sequence
from math import log, sqrt
from typing import Any

import numpy as np

from .cli import _write_profile_result, simulate_profiles
from .formula import parse_formula
from .isotopes import load_isotope_registry, split_formula_isotope_components

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except ImportError as exc:  # pragma: no cover - depends on Python build.
    raise RuntimeError("FastIso GUI requires tkinter to be installed") from exc


_FORMULA_SPLIT_RE = re.compile(r"[\s,;]+")
_DEFAULT_FORMULA = "C500H800N125O200S10"
_DEFAULT_PRESET = "full"
_DEFAULT_ELEMENTS = ""
_FWHM_TO_SIGMA = 2.0 * sqrt(2.0 * log(2.0))


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
        self.preset_var = tk.StringVar(value=_DEFAULT_PRESET)
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
        self.normalize_var = tk.StringVar(value="none")
        self.auto_window_sigma_var = tk.StringVar(value="6.0")
        self.auto_window_min_half_width_var = tk.StringVar(value="0.1")
        self.output_format_var = tk.StringVar(value="csv")
        self.status_var = tk.StringVar(value="Ready")
        self.show_peak_labels_var = tk.BooleanVar(value=True)

        self.result: dict[str, Any] | None = None
        self._worker: threading.Thread | None = None
        self._broadening_source = "rp"
        self._syncing_broadening = False
        self._plot_x_zoom = 1.0
        self._plot_y_zoom = 1.0
        self._plot_x_center: float | None = None
        self._plot_hover_tag = "plot-hover"
        self._plot_points: list[dict[str, float]] = []
        self._plot_geometry: dict[str, float] | None = None
        self._plot_drag_mode: str | None = None
        self._plot_drag_start: tuple[int, int] | None = None
        self._plot_drag_start_center: float | None = None
        self._plot_drag_start_y_zoom = 1.0

        self._install_broadening_traces()
        self._build_ui()
        self._sync_mode_state()
        self._sync_grid_state()
        self._sync_broadening_fields("rp")

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

        ttk.Label(controls, text="Normalize").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Combobox(
            controls,
            textvariable=self.normalize_var,
            values=("none", "sum", "max"),
            state="readonly",
            width=34,
        ).grid(row=row, column=1, sticky="ew", pady=4)
        row += 1

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

        plot_tools = ttk.Frame(output)
        plot_tools.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(2, 0))
        ttk.Button(plot_tools, text="X-", width=4, command=lambda: self._zoom_plot("x", 0.5)).pack(
            side="left",
            padx=(0, 4),
        )
        ttk.Button(plot_tools, text="X+", width=4, command=lambda: self._zoom_plot("x", 2.0)).pack(
            side="left",
            padx=(0, 8),
        )
        ttk.Button(plot_tools, text="Y-", width=4, command=lambda: self._zoom_plot("y", 0.5)).pack(
            side="left",
            padx=(0, 4),
        )
        ttk.Button(plot_tools, text="Y+", width=4, command=lambda: self._zoom_plot("y", 2.0)).pack(
            side="left",
            padx=(0, 8),
        )
        ttk.Button(plot_tools, text="Reset", width=7, command=self._reset_plot_view).pack(
            side="left",
            padx=(0, 12),
        )
        ttk.Checkbutton(
            plot_tools,
            text="Peak labels",
            variable=self.show_peak_labels_var,
            command=self._draw_plot,
        ).pack(side="left")

        self.plot = tk.Canvas(output, height=520, background="white", highlightthickness=1)
        self.plot.grid(row=3, column=0, columnspan=2, sticky="nsew", pady=(4, 0))
        self.plot.bind("<Configure>", lambda _event: self._draw_plot())
        self.plot.bind("<Motion>", self._on_plot_motion)
        self.plot.bind("<Leave>", lambda _event: self._hide_plot_tooltip())
        self.plot.bind("<MouseWheel>", self._on_plot_wheel)
        self.plot.bind("<ButtonPress-1>", self._on_plot_button_press)
        self.plot.bind("<B1-Motion>", self._on_plot_drag)
        self.plot.bind("<ButtonRelease-1>", self._on_plot_button_release)

        self.summary_text = tk.Text(output, height=8, wrap="word")
        self.summary_text.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(10, 0))

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

    def _install_broadening_traces(self) -> None:
        self.rp_var.trace_add("write", lambda *_args: self._on_broadening_edit("rp"))
        self.gaussian_sigma_var.trace_add(
            "write",
            lambda *_args: self._on_broadening_edit("sigma"),
        )
        for variable in (self.formula_var, self.preset_var, self.elements_var):
            variable.trace_add(
                "write",
                lambda *_args: self._sync_broadening_fields(self._broadening_source),
            )

    def _on_broadening_edit(self, source: str) -> None:
        if self._syncing_broadening:
            return
        self._broadening_source = source
        self._sync_broadening_fields(source)

    def _sync_broadening_fields(self, source: str) -> None:
        if self._syncing_broadening:
            return
        mean_mass = _single_formula_mean_mass(
            self.formula_var.get(),
            self.preset_var.get(),
            self.elements_var.get(),
        )
        self._syncing_broadening = True
        try:
            if source == "rp":
                resolving_power = _optional_float(self.rp_var.get(), "Resolving power")
                if mean_mass is None or resolving_power is None:
                    self.gaussian_sigma_var.set("")
                    return
                sigma = _sigma_from_resolving_power(mean_mass, resolving_power)
                self.gaussian_sigma_var.set(_format_float(sigma))
            elif source == "sigma":
                sigma = _optional_float(self.gaussian_sigma_var.get(), "Gaussian sigma")
                if mean_mass is None or sigma is None:
                    self.rp_var.set("")
                    return
                resolving_power = _resolving_power_from_sigma(mean_mass, sigma)
                self.rp_var.set(_format_float(resolving_power))
        except Exception:
            return
        finally:
            self._syncing_broadening = False

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
        self._clear_display()

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
        self._reset_plot_view(redraw=False)
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
        if self._broadening_source == "sigma" and gaussian_sigma is not None:
            resolving_power = None
        elif resolving_power is not None:
            gaussian_sigma = None

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
            "normalize": self.normalize_var.get().strip() or "none",
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
        self._draw_plot()
        self._render_summary(result)

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
            f"resolving power: {metadata.get('resolving_power')}",
            f"effective sigma: {_format_numeric_metadata(metadata.get('gaussian_sigma'))}",
            f"normalization: {metadata.get('normalization', 'none')}",
            "plot normalization: max",
            f"profile sum: {_format_numeric_metadata(metadata.get('profile_sums'))}",
            f"profile max: {_format_numeric_metadata(metadata.get('profile_maxima'))}",
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
        self._plot_points = []
        self._plot_geometry = None
        width = max(self.plot.winfo_width(), 2)
        height = max(self.plot.winfo_height(), 2)
        left_pad = 58
        right_pad = 14
        top_pad = 18
        bottom_pad = 36
        self.plot.create_rectangle(
            left_pad,
            top_pad,
            width - right_pad,
            height - bottom_pad,
            outline="#b0b0b0",
        )
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
        y_display = _max_normalized_intensity(y)
        x_min, x_max = self._plot_x_bounds(x)
        visible = (x >= x_min) & (x <= x_max)
        if not np.any(visible):
            return
        visible_x = x[visible]
        visible_y = y_display[visible]
        visible_raw = y[visible]
        plot_x, plot_y, plot_raw = _profile_plot_points_with_raw(
            visible_x,
            visible_y,
            visible_raw,
            max_points=2200,
        )
        y_min = min(0.0, float(np.min(visible_y)))
        y_max = max(1.0 / self._plot_y_zoom, float(np.max(visible_y)) if self._plot_y_zoom < 1.0 else 0.0)
        if y_max <= y_min:
            y_max = y_min + 1.0
        plot_w = width - left_pad - right_pad
        plot_h = height - top_pad - bottom_pad
        self._plot_geometry = {
            "x0": float(left_pad),
            "y0": float(top_pad),
            "x1": float(width - right_pad),
            "y1": float(height - bottom_pad),
            "x_min": float(x_min),
            "x_max": float(x_max),
            "full_x_min": float(np.min(x)),
            "full_x_max": float(np.max(x)),
            "y_min": float(y_min),
            "y_max": float(y_max),
        }
        coords: list[float] = []
        for x_val, y_val, raw_val in zip(plot_x, plot_y, plot_raw):
            px = left_pad + (float(x_val) - x_min) / (x_max - x_min) * plot_w
            py = top_pad + (1.0 - (float(y_val) - y_min) / (y_max - y_min)) * plot_h
            py = min(max(py, top_pad), height - bottom_pad)
            coords.extend((px, py))
            self._plot_points.append(
                {
                    "px": px,
                    "py": py,
                    "mass": float(x_val),
                    "norm": float(y_val),
                    "raw": float(raw_val),
                }
            )
        if len(coords) >= 4:
            self.plot.create_line(*coords, fill="#1f6feb", width=2)
        if y_min <= 0.0 <= y_max:
            baseline = top_pad + (1.0 - (0.0 - y_min) / (y_max - y_min)) * plot_h
            self.plot.create_line(left_pad, baseline, width - right_pad, baseline, fill="#808080")
        self._draw_plot_axes(
            left_pad,
            top_pad,
            width - right_pad,
            height - bottom_pad,
            x_min,
            x_max,
            y_min,
            y_max,
        )
        if self.show_peak_labels_var.get():
            self._draw_peak_labels(visible_x, visible_y, left_pad, top_pad, plot_w, plot_h, x_min, x_max, y_min, y_max)

    def _plot_x_bounds(self, x: np.ndarray) -> tuple[float, float]:
        full_min = float(np.min(x))
        full_max = float(np.max(x))
        if full_max == full_min:
            return full_min - 0.5, full_max + 0.5
        center = self._plot_x_center
        if center is None:
            center = 0.5 * (full_min + full_max)
        half_width = 0.5 * (full_max - full_min) / self._plot_x_zoom
        start = max(full_min, center - half_width)
        stop = min(full_max, center + half_width)
        if start == full_min:
            stop = min(full_max, start + 2.0 * half_width)
        if stop == full_max:
            start = max(full_min, stop - 2.0 * half_width)
        if stop <= start:
            stop = start + 1.0
        self._plot_x_center = 0.5 * (start + stop)
        return start, stop

    def _draw_plot_axes(
        self,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
    ) -> None:
        self.plot.create_text(x0, y1 + 18, text=_format_float(x_min), anchor="w", fill="#505050")
        self.plot.create_text(x1, y1 + 18, text=_format_float(x_max), anchor="e", fill="#505050")
        self.plot.create_text((x0 + x1) / 2, y1 + 18, text="mass", anchor="center", fill="#505050")
        self.plot.create_text(x0 - 8, y0, text=_format_float(y_max), anchor="e", fill="#505050")
        self.plot.create_text(x0 - 8, y1, text=_format_float(y_min), anchor="e", fill="#505050")
        self.plot.create_text(x0 - 42, (y0 + y1) / 2, text="max norm", anchor="center", angle=90, fill="#505050")

    def _draw_peak_labels(
        self,
        x: np.ndarray,
        y: np.ndarray,
        x0: int,
        y0: int,
        plot_w: int,
        plot_h: int,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
    ) -> None:
        peak_indices = _label_peak_indices(x, y, max_labels=12)
        used_px: list[float] = []
        for index in peak_indices:
            x_val = float(x[index])
            y_val = float(y[index])
            px = x0 + (x_val - x_min) / (x_max - x_min) * plot_w
            if any(abs(px - prev) < 36 for prev in used_px):
                continue
            py = y0 + (1.0 - (y_val - y_min) / (y_max - y_min)) * plot_h
            py = min(max(py, y0), y0 + plot_h)
            self.plot.create_line(px, py, px, max(y0, py - 10), fill="#606060")
            self.plot.create_text(
                px + 3,
                max(y0 + 6, py - 14),
                text=_format_peak_label(x_val),
                anchor="w",
                fill="#303030",
                font=("TkDefaultFont", 8),
            )
            used_px.append(px)

    def _zoom_plot(self, axis: str, factor: float) -> None:
        if axis == "x":
            self._plot_x_zoom = min(256.0, max(1.0, self._plot_x_zoom * float(factor)))
        elif axis == "y":
            self._plot_y_zoom = min(1024.0, max(0.125, self._plot_y_zoom * float(factor)))
        self._draw_plot()

    def _reset_plot_view(self, *, redraw: bool = True) -> None:
        self._plot_x_zoom = 1.0
        self._plot_y_zoom = 1.0
        self._plot_x_center = None
        self._hide_plot_tooltip()
        if redraw:
            self._draw_plot()

    def _on_plot_wheel(self, event: tk.Event) -> None:
        factor = 1.25 if event.delta > 0 else 0.8
        axis = "y" if event.state & 0x0001 else "x"
        self._zoom_plot(axis, factor)

    def _on_plot_motion(self, event: tk.Event) -> None:
        if self._plot_drag_mode is not None:
            return
        if not self._plot_points:
            return
        nearest = min(
            self._plot_points,
            key=lambda point: (point["px"] - event.x) ** 2 + (point["py"] - event.y) ** 2,
        )
        distance2 = (nearest["px"] - event.x) ** 2 + (nearest["py"] - event.y) ** 2
        if distance2 > 144:
            self._hide_plot_tooltip()
            return
        self._show_plot_tooltip(event.x, event.y, nearest)

    def _show_plot_tooltip(self, x: int, y: int, point: dict[str, float]) -> None:
        self._hide_plot_tooltip()
        text = (
            f"mass {point['mass']:.8f}\n"
            f"max norm {point['norm']:.6g}\n"
            f"raw {point['raw']:.6g}"
        )
        item = self.plot.create_text(
            x + 12,
            y - 12,
            text=text,
            anchor="sw",
            fill="#111111",
            tags=self._plot_hover_tag,
        )
        bbox = self.plot.bbox(item)
        if bbox is not None:
            rect = self.plot.create_rectangle(
                bbox[0] - 4,
                bbox[1] - 3,
                bbox[2] + 4,
                bbox[3] + 3,
                fill="#fff8c5",
                outline="#c8a600",
                tags=self._plot_hover_tag,
            )
            self.plot.tag_lower(rect, item)

    def _hide_plot_tooltip(self) -> None:
        self.plot.delete(self._plot_hover_tag)

    def _on_plot_button_press(self, event: tk.Event) -> None:
        mode = self._plot_drag_mode_at(event.x, event.y)
        if mode is None:
            return
        self._plot_drag_mode = mode
        self._plot_drag_start = (event.x, event.y)
        self._plot_drag_start_center = self._plot_x_center
        self._plot_drag_start_y_zoom = self._plot_y_zoom
        self._hide_plot_tooltip()
        self.plot.configure(cursor="sb_h_double_arrow" if mode == "x" else "sb_v_double_arrow")

    def _on_plot_drag(self, event: tk.Event) -> None:
        if self._plot_drag_mode is None or self._plot_drag_start is None:
            return
        geometry = self._plot_geometry
        if geometry is None:
            return
        start_x, start_y = self._plot_drag_start
        if self._plot_drag_mode == "x":
            start_center = (
                self._plot_drag_start_center
                if self._plot_drag_start_center is not None
                else 0.5 * (geometry["x_min"] + geometry["x_max"])
            )
            self._plot_x_center = _dragged_x_center(
                start_center,
                event.x - start_x,
                geometry["x1"] - geometry["x0"],
                geometry["full_x_min"],
                geometry["full_x_max"],
                geometry["x_max"] - geometry["x_min"],
            )
        elif self._plot_drag_mode == "y":
            self._plot_y_zoom = _dragged_y_zoom(
                self._plot_drag_start_y_zoom,
                event.y - start_y,
            )
        self._draw_plot()

    def _on_plot_button_release(self, _event: tk.Event) -> None:
        self._plot_drag_mode = None
        self._plot_drag_start = None
        self._plot_drag_start_center = None
        self.plot.configure(cursor="")

    def _plot_drag_mode_at(self, x: int, y: int) -> str | None:
        geometry = self._plot_geometry
        if geometry is None:
            return None
        if geometry["x0"] <= x <= geometry["x1"] and geometry["y1"] <= y <= geometry["y1"] + 30:
            return "x"
        if geometry["x0"] - 52 <= x <= geometry["x0"] and geometry["y0"] <= y <= geometry["y1"]:
            return "y"
        return None

    def _clear_display(self) -> None:
        self._plot_points = []
        self._plot_geometry = None
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


def _single_formula_mean_mass(
    formula_text: str,
    preset_text: str,
    elements_text: str,
) -> float | None:
    formulas = _parse_formula_list(formula_text)
    if len(formulas) != 1:
        return None
    preset = preset_text.strip() or "common"
    resource = "full" if preset == "full" else "common"
    registry = load_isotope_registry(resource)
    elements = _parse_elements(elements_text)
    selected_elements = (
        tuple(elements)
        if elements is not None
        else registry.elements_for_preset(preset)
    )
    component = split_formula_isotope_components(
        formulas[0],
        registry.patterns,
        elements=selected_elements,
    )
    spectral_mass = sum(
        count * registry.patterns[element].mean_mass
        for element, count in component.spectral_counts.items()
    )
    return float(spectral_mass + component.mass_shift)


def _sigma_from_resolving_power(mean_mass: float, resolving_power: float) -> float:
    if resolving_power <= 0.0:
        raise ValueError("resolving power must be positive")
    return float(mean_mass) / float(resolving_power) / _FWHM_TO_SIGMA


def _resolving_power_from_sigma(mean_mass: float, sigma: float) -> float:
    if sigma <= 0.0:
        raise ValueError("Gaussian sigma must be positive")
    return float(mean_mass) / (float(sigma) * _FWHM_TO_SIGMA)


def _format_float(value: float) -> str:
    return f"{float(value):.12g}"


def _format_peak_label(value: float) -> str:
    return f"{float(value):.3f}"


def _format_numeric_metadata(value: object) -> str:
    if value is None:
        return "None"
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        return _format_float(float(array))
    if array.size == 1:
        return _format_float(float(array[0]))
    return f"{_format_float(float(np.min(array)))}..{_format_float(float(np.max(array)))}"


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


def _profile_plot_points_with_raw(
    mass_axis: np.ndarray,
    display_intensity: np.ndarray,
    raw_intensity: np.ndarray,
    *,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected_mass, selected_display = _profile_plot_points(
        mass_axis,
        display_intensity,
        max_points=max_points,
    )
    indices = np.searchsorted(mass_axis, selected_mass)
    indices = np.clip(indices, 0, raw_intensity.size - 1)
    return selected_mass, selected_display, raw_intensity[indices]


def _dragged_x_center(
    start_center: float,
    dx_pixels: float,
    plot_width: float,
    full_x_min: float,
    full_x_max: float,
    visible_width: float,
) -> float:
    if plot_width <= 0.0 or visible_width <= 0.0:
        return float(start_center)
    full_width = float(full_x_max) - float(full_x_min)
    if full_width <= visible_width:
        return 0.5 * (float(full_x_min) + float(full_x_max))
    delta_mass = -float(dx_pixels) / float(plot_width) * float(visible_width)
    center = float(start_center) + delta_mass
    half_width = 0.5 * float(visible_width)
    return min(
        max(center, float(full_x_min) + half_width),
        float(full_x_max) - half_width,
    )


def _dragged_y_zoom(start_zoom: float, dy_pixels: float) -> float:
    factor = 2.0 ** (-float(dy_pixels) / 120.0)
    return min(1024.0, max(0.125, float(start_zoom) * factor))


def _max_normalized_intensity(intensity: np.ndarray) -> np.ndarray:
    y = np.asarray(intensity, dtype=np.float64)
    if y.size == 0:
        return y
    finite = y[np.isfinite(y)]
    if finite.size == 0:
        return np.zeros_like(y)
    maximum = float(np.max(finite))
    if maximum == 0.0:
        return np.zeros_like(y)
    return y / maximum


def _label_peak_indices(
    mass_axis: np.ndarray,
    intensity: np.ndarray,
    *,
    max_labels: int,
) -> np.ndarray:
    if max_labels <= 0 or mass_axis.size == 0:
        return np.array([], dtype=np.int64)
    local = _local_peak_indices(intensity)
    if local.size == 0:
        return np.array([], dtype=np.int64)
    order = np.argsort(intensity[local])[::-1]
    selected = local[order[:max_labels]]
    return np.array(sorted(selected), dtype=np.int64)


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
