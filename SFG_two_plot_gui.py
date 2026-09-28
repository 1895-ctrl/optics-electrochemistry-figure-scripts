"""Plot polarization-dependent SFG spectra and orientation responses.

The interface deliberately exposes only six physics controls, four plot-range
controls, and four polarization checkboxes.  The optical material is fixed to the CO/Pt(111)
literature benchmark used by ``sfg_two_plots.py``.

Run:
    python SFG_two_plot_gui.py

Self-test without opening the GUI:
    python SFG_two_plot_gui.py --selftest
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
from pathlib import Path
import math
import sys
from tempfile import TemporaryDirectory
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import zipfile
from xml.etree import ElementTree

import numpy as np

import sfg_two_plots as model


DEFAULTS = {
    "resonance": 2090.0,
    "hwhm": 8.0,
    "spectrum_tilt": 25.0,
    "ratio": 0.49,
    "visible_angle": 58.5,
    "ir_angle": 55.0,
    "frequency_min": 1800.0,
    "frequency_max": 2160.0,
    "tilt_min": 0.0,
    "tilt_max": 90.0,
}

POLARIZATIONS = ("ppp", "ssp", "sps", "pss")
SPECTRUM_POINTS = 721
TILT_POINTS = 361


# These are loaded only when the GUI starts, so --selftest does not require
# matplotlib or a display server.
Figure = None
FigureCanvasTkAgg = None
NavigationToolbar2Tk = None


@dataclass(frozen=True)
class UIParameters:
    resonance: float
    hwhm: float
    spectrum_tilt: float
    ratio: float
    visible_angle: float
    ir_angle: float
    frequency_min: float
    frequency_max: float
    tilt_min: float
    tilt_max: float
    polarizations: tuple[str, ...]


def apply_parameters(params: UIParameters) -> None:
    """Apply validated UI values to the compact calculation module."""

    model.RESONANCE_CM1 = params.resonance
    model.HWHM_CM1 = params.hwhm
    model.SPECTRUM_TILT_DEG = params.spectrum_tilt
    model.HYPERPOLARIZABILITY_RATIO = params.ratio
    model.VISIBLE_INCIDENCE_DEG = params.visible_angle
    model.IR_INCIDENCE_DEG = params.ir_angle


def upright_ratio(resonance_cm1: float) -> float:
    """Return I_PPP/I_SSP for an upright molecule at the current settings."""

    i_ppp = float(model.intensity(resonance_cm1, 0.0, "ppp"))
    i_ssp = float(model.intensity(resonance_cm1, 0.0, "ssp"))
    return i_ppp / i_ssp if i_ssp > np.finfo(float).tiny else math.inf


def calculate_spectrum_data(params: UIParameters):
    """Return raw and plotted spectrum data using one common normalization."""

    wavenumber = np.linspace(
        params.frequency_min, params.frequency_max, SPECTRUM_POINTS
    )
    raw = {
        pol: np.asarray(
            model.intensity(wavenumber, params.spectrum_tilt, pol), dtype=float
        )
        for pol in params.polarizations
    }
    common_scale = max(float(np.max(values)) for values in raw.values())
    common_scale = max(common_scale, np.finfo(float).tiny)
    normalized = {pol: values / common_scale for pol, values in raw.items()}
    return wavenumber, raw, normalized, common_scale


def calculate_tilt_data(params: UIParameters):
    """Return peak intensity at the resonance versus tilt angle."""

    tilt = np.linspace(params.tilt_min, params.tilt_max, TILT_POINTS)
    raw = {
        pol: np.asarray(model.intensity(params.resonance, tilt, pol), dtype=float)
        for pol in params.polarizations
    }
    normalized = {}
    own_maxima = {}
    for pol, values in raw.items():
        own_maximum = max(float(np.max(values)), np.finfo(float).tiny)
        own_maxima[pol] = own_maximum
        normalized[pol] = values / own_maximum
    return tilt, raw, normalized, own_maxima


def _xlsx_col_name(index: int) -> str:
    index += 1
    out = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        out = chr(65 + remainder) + out
    return out


def _xlsx_cell(ref: str, value, header: bool = False, numeric_style: int = 2) -> str:
    if header:
        style = ' s="1"'
    else:
        style = f' s="{numeric_style}"' if isinstance(
            value, (int, float, np.integer, np.floating)
        ) and not isinstance(value, (bool, np.bool_)) else ""

    if value is None:
        return f'<c r="{ref}"{style}/>'
    if isinstance(value, (bool, np.bool_)):
        return f'<c r="{ref}" t="b"{style}><v>{1 if value else 0}</v></c>'
    if isinstance(value, (int, float, np.integer, np.floating)):
        if not math.isfinite(float(value)):
            return f'<c r="{ref}"{style}/>'
        return f'<c r="{ref}"{style}><v>{float(value):.15g}</v></c>'
    text = escape(str(value))
    return (
        f'<c r="{ref}" t="inlineStr"{style}><is>'
        f'<t xml:space="preserve">{text}</t></is></c>'
    )


def _xlsx_sheet_xml(rows, decimal_columns=(), scientific_cells=()) -> str:
    """Build one simple, filterable worksheet with a frozen header row."""

    if not rows:
        rows = [[""]]
    row_count = len(rows)
    column_count = max(1, max(len(row) for row in rows))
    widths = []
    for column in range(column_count):
        samples = [
            str(row[column])
            for row in rows[:201]
            if column < len(row) and row[column] is not None
        ]
        widths.append(min(max([12] + [len(value) + 2 for value in samples]), 34))
    columns_xml = "".join(
        f'<col min="{index + 1}" max="{index + 1}" width="{width}" customWidth="1"/>'
        for index, width in enumerate(widths)
    )

    row_xml = []
    for row_index, row in enumerate(rows, start=1):
        cells = []
        for column_index, value in enumerate(row):
            if (row_index, column_index) in scientific_cells:
                numeric_style = 2
            else:
                numeric_style = 3 if column_index in decimal_columns else 2
            cells.append(
                _xlsx_cell(
                    f"{_xlsx_col_name(column_index)}{row_index}",
                    value,
                    header=row_index == 1,
                    numeric_style=numeric_style,
                )
            )
        height = ' ht="24" customHeight="1"' if row_index == 1 else ""
        row_xml.append(f'<row r="{row_index}"{height}>{"".join(cells)}</row>')

    dimension = f"A1:{_xlsx_col_name(column_count - 1)}{row_count}"
    auto_filter = f'<autoFilter ref="{dimension}"/>' if row_count > 1 else ""
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<dimension ref="{dimension}"/>'
        '<sheetViews><sheetView workbookViewId="0" showGridLines="0">'
        '<pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/>'
        '</sheetView></sheetViews><sheetFormatPr defaultRowHeight="18"/>'
        f'<cols>{columns_xml}</cols><sheetData>{"".join(row_xml)}</sheetData>'
        f'{auto_filter}</worksheet>'
    )


def write_xlsx_export(path: Path, params: UIParameters) -> None:
    """Export raw and normalized data for the two displayed SFG plots."""

    wavenumber, spectrum_raw, spectrum_norm, spectrum_scale = (
        calculate_spectrum_data(params)
    )
    tilt, tilt_raw, tilt_norm, tilt_maxima = calculate_tilt_data(params)

    spectrum_columns = ["IR wavenumber (cm^-1)"]
    for pol in params.polarizations:
        spectrum_columns.extend(
            [f"{pol.upper()} raw intensity (a.u.)", f"{pol.upper()} normalized intensity"]
        )
    spectrum_rows = [spectrum_columns]
    for index, axis_value in enumerate(wavenumber):
        row = [float(axis_value)]
        for pol in params.polarizations:
            row.extend([float(spectrum_raw[pol][index]), float(spectrum_norm[pol][index])])
        spectrum_rows.append(row)

    tilt_columns = ["Tilt angle from surface normal (deg)"]
    for pol in params.polarizations:
        tilt_columns.extend(
            [
                f"{pol.upper()} raw peak intensity (a.u.)",
                f"{pol.upper()} normalized peak intensity",
            ]
        )
    tilt_rows = [tilt_columns]
    for index, axis_value in enumerate(tilt):
        row = [float(axis_value)]
        for pol in params.polarizations:
            row.extend([float(tilt_raw[pol][index]), float(tilt_norm[pol][index])])
        tilt_rows.append(row)

    parameter_rows = [
        ["Parameter", "Value", "Unit or definition"],
        ["Model", "CO/Pt(111)", "one resonant mode; no NR background"],
        ["Visible wavelength", model.VISIBLE_WAVELENGTH_NM, "nm"],
        ["Peak center", params.resonance, "cm^-1"],
        ["HWHM", params.hwhm, "cm^-1"],
        ["FWHM", 2.0 * params.hwhm, "cm^-1"],
        ["Tilt used in wavenumber plot", params.spectrum_tilt, "deg from surface normal"],
        ["R", params.ratio, "beta_aac / beta_ccc"],
        ["Visible incidence angle", params.visible_angle, "deg from surface normal"],
        ["IR incidence angle", params.ir_angle, "deg from surface normal"],
        ["Wavenumber minimum", params.frequency_min, "cm^-1"],
        ["Wavenumber maximum", params.frequency_max, "cm^-1"],
        ["Tilt minimum", params.tilt_min, "deg"],
        ["Tilt maximum", params.tilt_max, "deg"],
        ["Polarizations", ", ".join(pol.upper() for pol in params.polarizations), "order: SFG, VIS, IR"],
        ["Spectrum normalization", spectrum_scale, "common maximum across selected polarizations"],
        [
            "Tilt normalization",
            "; ".join(f"{pol.upper()}={tilt_maxima[pol]:.8e}" for pol in params.polarizations),
            "each polarization divided by its own maximum",
        ],
        ["Generated at", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"), ""],
        ["Literature benchmark", "Li et al., Topics in Catalysis 61, 751-762 (2018)", "doi:10.1007/s11244-018-0949-7"],
    ]

    sheets = [
        ("Intensity vs wavenumber", spectrum_rows, {0}, set()),
        ("Peak intensity vs tilt angle", tilt_rows, {0}, set()),
        ("Parameters", parameter_rows, {1}, {(16, 1)}),
    ]

    sheet_overrides = "".join(
        f'<Override PartName="/xl/worksheets/sheet{index}.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        for index in range(1, len(sheets) + 1)
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/styles.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
        '<Override PartName="/docProps/core.xml" '
        'ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>'
        '<Override PartName="/docProps/app.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>'
        f'{sheet_overrides}</Types>'
    )
    package_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/>'
        '<Relationship Id="rId2" '
        'Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" '
        'Target="docProps/core.xml"/>'
        '<Relationship Id="rId3" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" '
        'Target="docProps/app.xml"/></Relationships>'
    )
    workbook_sheets = "".join(
        f'<sheet name="{escape(name)}" sheetId="{index}" r:id="rId{index}"/>'
        for index, (name, _, _, _) in enumerate(sheets, start=1)
    )
    workbook_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'<bookViews><workbookView/></bookViews><sheets>{workbook_sheets}</sheets>'
        '<calcPr calcId="0"/></workbook>'
    )
    workbook_rels = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">',
    ]
    for index in range(1, len(sheets) + 1):
        workbook_rels.append(
            f'<Relationship Id="rId{index}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            f'Target="worksheets/sheet{index}.xml"/>'
        )
    workbook_rels.append(
        f'<Relationship Id="rId{len(sheets) + 1}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
        'Target="styles.xml"/></Relationships>'
    )
    styles_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<numFmts count="2">'
        '<numFmt numFmtId="164" formatCode="0.00000000E+00"/>'
        '<numFmt numFmtId="165" formatCode="0.000"/>'
        '</numFmts>'
        '<fonts count="2"><font><sz val="10"/><name val="Aptos"/></font>'
        '<font><b/><color rgb="FFFFFFFF"/><sz val="10"/><name val="Aptos"/></font></fonts>'
        '<fills count="3"><fill><patternFill patternType="none"/></fill>'
        '<fill><patternFill patternType="gray125"/></fill>'
        '<fill><patternFill patternType="solid"><fgColor rgb="FF0072B2"/>'
        '<bgColor indexed="64"/></patternFill></fill></fills>'
        '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
        '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
        '<cellXfs count="4">'
        '<xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>'
        '<xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" applyFont="1" applyFill="1" applyAlignment="1"><alignment horizontal="center" vertical="center"/></xf>'
        '<xf numFmtId="164" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/>'
        '<xf numFmtId="165" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/>'
        '</cellXfs><cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/>'
        '</cellStyles></styleSheet>'
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    core_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/" '
        'xmlns:dcterms="http://purl.org/dc/terms/" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
        '<dc:creator>sfg_two_plot_gui.py</dc:creator>'
        '<dc:title>SFG simulation data</dc:title>'
        f'<dcterms:created xsi:type="dcterms:W3CDTF">{timestamp}</dcterms:created>'
        '</cp:coreProperties>'
    )
    app_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" '
        'xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">'
        '<Application>Minimal SFG GUI</Application></Properties>'
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", package_rels)
        archive.writestr("docProps/core.xml", core_xml)
        archive.writestr("docProps/app.xml", app_xml)
        archive.writestr("xl/workbook.xml", workbook_xml)
        archive.writestr("xl/_rels/workbook.xml.rels", "".join(workbook_rels))
        archive.writestr("xl/styles.xml", styles_xml)
        for index, (_, rows, decimal_columns, scientific_cells) in enumerate(
            sheets, start=1
        ):
            archive.writestr(
                f"xl/worksheets/sheet{index}.xml",
                _xlsx_sheet_xml(rows, decimal_columns, scientific_cells),
            )


class MinimalSFGApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("SFG: polarization spectrum and molecular tilt")
        self.root.geometry("1240x780")
        self.root.minsize(980, 650)

        self.values = {
            key: tk.StringVar(value=f"{value:g}")
            for key, value in DEFAULTS.items()
        }
        self.pol_enabled = {
            pol: tk.BooleanVar(value=True) for pol in POLARIZATIONS
        }
        self.status_text = tk.StringVar(value="Ready")

        self._build_controls()
        self._build_plots()
        self.update_plots()

    def _build_controls(self) -> None:
        controls = ttk.Frame(self.root, padding=12)
        controls.pack(side="left", fill="y")

        ttk.Label(
            controls,
            text="Minimal SFG controls",
            font=("TkDefaultFont", 13, "bold"),
        ).pack(anchor="w", pady=(0, 8))
        ttk.Label(
            controls,
            text=(
                "Fixed model: CO/Pt(111), 532 nm,\n"
                "one resonant mode, no NR background."
            ),
            foreground="#555555",
            justify="left",
        ).pack(anchor="w", pady=(0, 10))

        physics = ttk.LabelFrame(controls, text="Adjustable parameters", padding=10)
        physics.pack(fill="x")
        fields = (
            ("resonance", "Peak center (cm⁻¹)"),
            ("hwhm", "HWHM (cm⁻¹)"),
            ("spectrum_tilt", "Tilt in spectrum (deg)"),
            ("ratio", "R = βaac / βccc"),
            ("visible_angle", "Visible angle (deg)"),
            ("ir_angle", "IR angle (deg)"),
        )
        for row, (key, label) in enumerate(fields):
            ttk.Label(physics, text=label).grid(
                row=row, column=0, sticky="w", padx=(0, 8), pady=4
            )
            entry = ttk.Entry(physics, textvariable=self.values[key], width=10)
            entry.grid(row=row, column=1, sticky="ew", pady=4)
            entry.bind("<Return>", lambda _event: self.update_plots())
        physics.columnconfigure(1, weight=1)

        ranges = ttk.LabelFrame(controls, text="Plot ranges", padding=10)
        ranges.pack(fill="x", pady=(10, 0))
        range_fields = (
            ("frequency_min", "Frequency min (cm⁻¹)"),
            ("frequency_max", "Frequency max (cm⁻¹)"),
            ("tilt_min", "Tilt min (deg)"),
            ("tilt_max", "Tilt max (deg)"),
        )
        for row, (key, label) in enumerate(range_fields):
            ttk.Label(ranges, text=label).grid(
                row=row, column=0, sticky="w", padx=(0, 8), pady=4
            )
            entry = ttk.Entry(ranges, textvariable=self.values[key], width=10)
            entry.grid(row=row, column=1, sticky="ew", pady=4)
            entry.bind("<Return>", lambda _event: self.update_plots())
        ranges.columnconfigure(1, weight=1)

        pol_frame = ttk.LabelFrame(controls, text="Polarization combinations", padding=10)
        pol_frame.pack(fill="x", pady=(10, 0))
        for index, pol in enumerate(POLARIZATIONS):
            ttk.Checkbutton(
                pol_frame,
                text=pol.upper(),
                variable=self.pol_enabled[pol],
                command=self.update_plots,
            ).grid(row=index // 2, column=index % 2, sticky="w", padx=(0, 22), pady=3)

        button_row = ttk.Frame(controls)
        button_row.pack(fill="x", pady=(12, 0))
        ttk.Button(
            button_row,
            text="Update plots",
            command=self.update_plots,
        ).pack(fill="x")
        ttk.Button(
            button_row,
            text="Reset literature values",
            command=self.reset_defaults,
        ).pack(fill="x", pady=(6, 0))
        ttk.Button(
            button_row,
            text="Save two PNGs",
            command=self.save_plots,
        ).pack(fill="x", pady=(6, 0))
        ttk.Button(
            button_row,
            text="Export data to Excel",
            command=self.export_excel,
        ).pack(fill="x", pady=(6, 0))

        ttk.Separator(controls).pack(fill="x", pady=12)
        ttk.Label(
            controls,
            textvariable=self.status_text,
            wraplength=260,
            justify="left",
        ).pack(anchor="w")

    def _build_plots(self) -> None:
        plot_area = ttk.Frame(self.root, padding=(0, 10, 10, 10))
        plot_area.pack(side="right", fill="both", expand=True)

        notebook = ttk.Notebook(plot_area)
        notebook.pack(fill="both", expand=True)

        spectrum_tab = ttk.Frame(notebook)
        tilt_tab = ttk.Frame(notebook)
        notebook.add(spectrum_tab, text="1  SFG intensity vs wavenumber")
        notebook.add(tilt_tab, text="2  SFG peak intensity vs tilt angle")

        self.spectrum_figure = Figure(figsize=(8.8, 6.6), dpi=100)
        self.spectrum_axis = self.spectrum_figure.add_subplot(111)
        self.spectrum_canvas = FigureCanvasTkAgg(
            self.spectrum_figure, master=spectrum_tab
        )
        self.spectrum_canvas.get_tk_widget().pack(fill="both", expand=True)
        spectrum_toolbar = NavigationToolbar2Tk(
            self.spectrum_canvas, spectrum_tab, pack_toolbar=False
        )
        spectrum_toolbar.update()
        spectrum_toolbar.pack(fill="x")

        self.tilt_figure = Figure(figsize=(8.8, 6.6), dpi=100)
        self.tilt_axis = self.tilt_figure.add_subplot(111)
        self.tilt_canvas = FigureCanvasTkAgg(self.tilt_figure, master=tilt_tab)
        self.tilt_canvas.get_tk_widget().pack(fill="both", expand=True)
        tilt_toolbar = NavigationToolbar2Tk(
            self.tilt_canvas, tilt_tab, pack_toolbar=False
        )
        tilt_toolbar.update()
        tilt_toolbar.pack(fill="x")

    def read_parameters(self) -> UIParameters:
        try:
            numeric = {key: float(variable.get()) for key, variable in self.values.items()}
        except ValueError as error:
            raise ValueError("All parameters and plot ranges must be valid numbers.") from error

        if not (500.0 <= numeric["resonance"] <= 5000.0):
            raise ValueError("Peak center must be between 500 and 5000 cm⁻¹.")
        if not (0.1 <= numeric["hwhm"] <= 200.0):
            raise ValueError("HWHM must be between 0.1 and 200 cm⁻¹.")
        if not (0.0 <= numeric["spectrum_tilt"] <= 90.0):
            raise ValueError("Spectrum tilt must be between 0 and 90 degrees.")
        if not (0.0 <= numeric["ratio"] <= 1.5):
            raise ValueError("R must be between 0 and 1.5.")
        if not (0.0 < numeric["visible_angle"] < 89.0):
            raise ValueError("Visible incidence angle must lie between 0 and 89 degrees.")
        if not (0.0 < numeric["ir_angle"] < 89.0):
            raise ValueError("IR incidence angle must lie between 0 and 89 degrees.")
        if not (1.0 <= numeric["frequency_min"] < numeric["frequency_max"]):
            raise ValueError("Frequency minimum must be positive and below the maximum.")
        if numeric["frequency_max"] > 10000.0:
            raise ValueError("Frequency maximum must not exceed 10000 cm⁻¹.")
        if not (numeric["frequency_min"] <= numeric["resonance"]
                <= numeric["frequency_max"]):
            raise ValueError("The frequency range must include the resonance center.")
        if not (0.0 <= numeric["tilt_min"] < numeric["tilt_max"] <= 90.0):
            raise ValueError("Tilt range must satisfy 0 ≤ min < max ≤ 90 degrees.")
        if not (numeric["tilt_min"] <= numeric["spectrum_tilt"]
                <= numeric["tilt_max"]):
            raise ValueError("The spectrum tilt must lie inside the selected tilt range.")

        selected = tuple(
            pol for pol in POLARIZATIONS if self.pol_enabled[pol].get()
        )
        if not selected:
            raise ValueError("Select at least one polarization combination.")

        return UIParameters(
            resonance=numeric["resonance"],
            hwhm=numeric["hwhm"],
            spectrum_tilt=numeric["spectrum_tilt"],
            ratio=numeric["ratio"],
            visible_angle=numeric["visible_angle"],
            ir_angle=numeric["ir_angle"],
            frequency_min=numeric["frequency_min"],
            frequency_max=numeric["frequency_max"],
            tilt_min=numeric["tilt_min"],
            tilt_max=numeric["tilt_max"],
            polarizations=selected,
        )

    def update_plots(self) -> None:
        try:
            params = self.read_parameters()
            apply_parameters(params)
            ratio = upright_ratio(params.resonance)
            self._draw_spectrum(params)
            self._draw_tilt(params, ratio)
            self.status_text.set(
                "Updated. Current upright ratio: "
                f"I_PPP/I_SSP = {ratio:.2f}.\n"
                "Literature reference: 27 at the reset values."
            )
        except Exception as error:
            self.status_text.set(f"Input error: {error}")
            messagebox.showerror("Invalid SFG settings", str(error), parent=self.root)

    def _draw_spectrum(self, params: UIParameters) -> None:
        wavenumber, _raw, curves, _common_scale = calculate_spectrum_data(params)

        ax = self.spectrum_axis
        ax.clear()
        for pol, values in curves.items():
            color, linestyle = model.PLOT_STYLE[pol]
            ax.plot(
                wavenumber,
                np.maximum(values, 1.0e-9),
                color=color,
                linestyle=linestyle,
                linewidth=model.PLOT_LINE_WIDTH,
                label=pol.upper(),
            )
        if wavenumber[0] <= model.LITERATURE_PEAK_CM1 <= wavenumber[-1]:
            ax.axvline(
                model.LITERATURE_PEAK_CM1,
                color="0.35",
                linestyle=(0, (2, 3)),
                linewidth=1.1,
                label=r"Li et al.: 2092 cm$^{-1}$",
            )
        ax.set_yscale("log")
        ax.set_xlim(params.frequency_min, params.frequency_max)
        ax.set_ylim(1.0e-7, 1.5)
        ax.set_xlabel(
            r"IR wavenumber (cm$^{-1}$)",
            fontsize=model.PLOT_AXIS_LABEL_SIZE,
            labelpad=7,
        )
        ax.set_ylabel(
            "Normalized SFG intensity",
            fontsize=model.PLOT_AXIS_LABEL_SIZE,
            labelpad=7,
        )
        ax.set_title(
            f"SFG intensity vs wavenumber  (tilt = {params.spectrum_tilt:g} deg)",
            fontsize=model.PLOT_TITLE_SIZE,
            pad=10,
        )
        model.apply_publication_axis_style(ax)
        ax.legend(
            fontsize=model.PLOT_LEGEND_SIZE,
            ncol=2,
            frameon=True,
            framealpha=0.94,
            borderpad=0.45,
            handlelength=2.7,
        )
        self.spectrum_figure.tight_layout(pad=1.1)
        self.spectrum_canvas.draw_idle()

    def _draw_tilt(self, params: UIParameters, ratio: float) -> None:
        tilt, _raw, curves, _own_maxima = calculate_tilt_data(params)
        ax = self.tilt_axis
        ax.clear()
        for pol in params.polarizations:
            color, linestyle = model.PLOT_STYLE[pol]
            ax.plot(
                tilt,
                curves[pol],
                color=color,
                linestyle=linestyle,
                linewidth=model.PLOT_LINE_WIDTH,
                label=pol.upper(),
            )
        ax.text(
            0.98,
            0.97,
            f"Current upright $I_{{PPP}}/I_{{SSP}}$ = {ratio:.2f}\n"
            "Li et al. reference = 27",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=model.PLOT_ANNOTATION_SIZE,
            bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.9},
        )
        ax.set_xlim(params.tilt_min, params.tilt_max)
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel(
            "Tilt angle from the surface normal (deg)",
            fontsize=model.PLOT_AXIS_LABEL_SIZE,
            labelpad=7,
        )
        ax.set_ylabel(
            "Normalized SFG peak intensity",
            fontsize=model.PLOT_AXIS_LABEL_SIZE,
            labelpad=7,
        )
        ax.set_title(
            "SFG peak intensity vs tilt angle",
            fontsize=model.PLOT_TITLE_SIZE,
            pad=10,
        )
        model.apply_publication_axis_style(ax)
        ax.legend(
            fontsize=model.PLOT_LEGEND_SIZE,
            ncol=2,
            loc="lower right",
            frameon=True,
            framealpha=0.94,
            borderpad=0.45,
            handlelength=2.7,
        )
        self.tilt_figure.tight_layout(pad=1.1)
        self.tilt_canvas.draw_idle()

    def reset_defaults(self) -> None:
        for key, value in DEFAULTS.items():
            self.values[key].set(f"{value:g}")
        for variable in self.pol_enabled.values():
            variable.set(True)
        self.update_plots()

    def save_plots(self) -> None:
        try:
            params = self.read_parameters()
            apply_parameters(params)
            output_dir = Path(__file__).resolve().parent / "sfg_two_plots_output"
            output_dir.mkdir(parents=True, exist_ok=True)
            frequency_path = output_dir / "sfg_intensity_vs_wavenumber.png"
            tilt_path = output_dir / "sfg_peak_intensity_vs_tilt_angle.png"
            self.spectrum_figure.savefig(frequency_path, dpi=300, facecolor="white")
            self.tilt_figure.savefig(tilt_path, dpi=300, facecolor="white")
            self.status_text.set(
                "Saved two PNGs to:\n"
                f"{output_dir}"
            )
        except Exception as error:
            messagebox.showerror("Could not save figures", str(error), parent=self.root)

    def export_excel(self) -> None:
        try:
            params = self.read_parameters()
            apply_parameters(params)
            output_dir = Path(__file__).resolve().parent / "sfg_two_plots_output"
            output_dir.mkdir(parents=True, exist_ok=True)
            selected_path = filedialog.asksaveasfilename(
                parent=self.root,
                title="Export SFG data to Excel",
                initialdir=str(output_dir),
                initialfile="sfg_two_plot_data.xlsx",
                defaultextension=".xlsx",
                filetypes=[("Excel workbook", "*.xlsx")],
            )
            if not selected_path:
                self.status_text.set("Excel export cancelled.")
                return
            output_path = Path(selected_path)
            write_xlsx_export(output_path, params)
            self.status_text.set(f"Exported Excel data to:\n{output_path}")
            messagebox.showinfo(
                "Excel export complete",
                "Saved raw and normalized data for both plots to:\n"
                f"{output_path}",
                parent=self.root,
            )
        except Exception as error:
            self.status_text.set(f"Excel export error: {error}")
            messagebox.showerror("Could not export Excel data", str(error), parent=self.root)


def run_selftest() -> bool:
    params = UIParameters(
        resonance=DEFAULTS["resonance"],
        hwhm=DEFAULTS["hwhm"],
        spectrum_tilt=DEFAULTS["spectrum_tilt"],
        ratio=DEFAULTS["ratio"],
        visible_angle=DEFAULTS["visible_angle"],
        ir_angle=DEFAULTS["ir_angle"],
        frequency_min=DEFAULTS["frequency_min"],
        frequency_max=DEFAULTS["frequency_max"],
        tilt_min=DEFAULTS["tilt_min"],
        tilt_max=DEFAULTS["tilt_max"],
        polarizations=POLARIZATIONS,
    )
    apply_parameters(params)
    ratio = upright_ratio(params.resonance)
    frequency = np.linspace(params.frequency_min, params.frequency_max, 51)
    tilt = np.linspace(params.tilt_min, params.tilt_max, 31)
    spectra_ok = all(
        model.intensity(frequency, params.spectrum_tilt, pol).shape == frequency.shape
        for pol in POLARIZATIONS
    )
    tilt_ok = all(
        model.intensity(params.resonance, tilt, pol).shape == tilt.shape
        for pol in POLARIZATIONS
    )
    ratio_ok = abs(ratio - 27.0) / 27.0 < 0.02

    with TemporaryDirectory() as temporary_directory:
        workbook_path = Path(temporary_directory) / "selftest_sfg_data.xlsx"
        write_xlsx_export(workbook_path, params)
        required_members = {
            "[Content_Types].xml",
            "xl/workbook.xml",
            "xl/styles.xml",
            "xl/worksheets/sheet1.xml",
            "xl/worksheets/sheet2.xml",
            "xl/worksheets/sheet3.xml",
        }
        with zipfile.ZipFile(workbook_path, "r") as archive:
            members_ok = required_members.issubset(archive.namelist())
            xml_ok = True
            for member in required_members:
                if member.endswith(".xml"):
                    try:
                        ElementTree.fromstring(archive.read(member))
                    except ElementTree.ParseError:
                        xml_ok = False
                        break
            workbook_xml = archive.read("xl/workbook.xml").decode("utf-8")
            names_ok = all(
                name in workbook_xml
                for name in (
                    "Intensity vs wavenumber",
                    "Peak intensity vs tilt angle",
                    "Parameters",
                )
            )
        xlsx_ok = workbook_path.stat().st_size > 1000 and members_ok and xml_ok and names_ok

    passed = spectra_ok and tilt_ok and ratio_ok and xlsx_ok
    print(
        f"[{'PASS' if passed else 'FAIL'}] minimal GUI model: "
        f"ratio={ratio:.4f}, spectra={spectra_ok}, tilt={tilt_ok}, "
        f"xlsx={xlsx_ok}"
    )
    return passed


def launch_gui() -> None:
    global Figure, FigureCanvasTkAgg, NavigationToolbar2Tk

    import matplotlib

    matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import (
        FigureCanvasTkAgg as Canvas,
        NavigationToolbar2Tk as Toolbar,
    )
    from matplotlib.figure import Figure as MatplotlibFigure

    Figure = MatplotlibFigure
    FigureCanvasTkAgg = Canvas
    NavigationToolbar2Tk = Toolbar

    root = tk.Tk()
    try:
        ttk.Style().theme_use("clam")
    except tk.TclError:
        pass
    MinimalSFGApp(root)
    root.mainloop()


def main() -> int:
    if "--selftest" in sys.argv:
        return 0 if run_selftest() else 1
    launch_gui()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
