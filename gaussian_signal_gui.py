"""Draw two Gaussian components and their pointwise sum.

The GUI previews the curves and exports transparent PNG or editable SVG files.
These curves are illustrative signals, not fits to experimental data.

Dependencies: Python 3 with Tkinter, Pillow and NumPy.

Run the GUI:
    python gaussian_signal_gui.py

Run a non-interactive export check:
    python gaussian_signal_gui.py --smoke-test --output gaussian_test.png
"""

from __future__ import annotations

import argparse
import math
from html import escape
import tkinter as tk
from dataclasses import dataclass, replace
from pathlib import Path
from tkinter import colorchooser, filedialog, messagebox, ttk

try:
    import numpy as np
    from PIL import Image, ImageColor, ImageDraw, ImageFont, ImageTk
except ImportError as exc:  # pragma: no cover - user-facing dependency message
    raise SystemExit(
        "This GUI requires numpy and Pillow. Install them with: pip install numpy pillow"
    ) from exc


APP_DIR = Path(__file__).resolve().parent
RESAMPLING = getattr(Image, "Resampling", Image)


@dataclass(frozen=True)
class PlotSettings:
    separation: float = 2.5
    sigma1: float = 1.0
    sigma2: float = 1.0
    amplitude1: float = 1.0
    amplitude2: float = 1.0
    baseline: float = 0.0
    x_half_range: float = 6.0
    component_width: float = 1.5
    sum_width: float = 2.4
    component1_color: str = "#7F7F7F"
    component2_color: str = "#A0A0A0"
    sum_color: str = "#111111"
    show_axes: bool = True
    show_legend: bool = True
    show_grid: bool = False
    show_centers: bool = False
    show_baseline: bool = False
    normalize: bool = False
    xlabel: str = "Position"
    ylabel: str = "Signal"
    title: str = ""
    figure_width: float = 5.6
    figure_height: float = 3.7
    dpi: int = 300


def gaussian(x: np.ndarray, center: float, sigma: float, amplitude: float) -> np.ndarray:
    """Return a Gaussian with peak value ``amplitude``."""
    return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def calculate_curves(settings: PlotSettings) -> tuple[np.ndarray, ...]:
    """Calculate both components, their sum, and the background baseline."""
    x = np.linspace(-settings.x_half_range, settings.x_half_range, 2401)
    center1 = -settings.separation / 2.0
    center2 = settings.separation / 2.0
    g1 = gaussian(x, center1, settings.sigma1, settings.amplitude1)
    g2 = gaussian(x, center2, settings.sigma2, settings.amplitude2)
    total = settings.baseline + g1 + g2
    baseline = np.full_like(x, settings.baseline)

    if settings.normalize:
        scale = float(np.max(total))
        if scale > 0:
            g1 = g1 / scale
            g2 = g2 / scale
            total = total / scale
            baseline = baseline / scale
    return x, g1, g2, total, baseline


def export_svg_file(settings: PlotSettings, output: Path) -> None:
    """Write transparent vector curves and editable text from the GUI parameters."""
    w = max(320, round(settings.figure_width * settings.dpi))
    h = max(220, round(settings.figure_height * settings.dpi))
    left, right = (68, w-20) if settings.show_axes else (8, w-8)
    top, bottom = ((46 if settings.title.strip() else 22), h-58) if settings.show_axes else (8,h-8)
    x, g1, g2, total, baseline = calculate_curves(settings)
    hi = max(float(np.max(v)) for v in (g1,g2,total))
    lo = min(0.,float(np.min(total)),float(np.min(baseline)))
    span = max(hi-lo,.1)
    ymin, ymax = lo-.015*span, hi+.08*span
    def px(v): return left+(v+settings.x_half_range)/(2*settings.x_half_range)*(right-left)
    def py(v): return bottom-(v-ymin)/(ymax-ymin)*(bottom-top)
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{settings.figure_width}in" height="{settings.figure_height}in" viewBox="0 0 {w} {h}">', '<title>Gaussian signals</title>']
    def line(x1,y1,x2,y2,color='#111111',width=1,dash=''):
        parts.append(f'<path d="M{x1:.4f},{y1:.4f} L{x2:.4f},{y2:.4f}" fill="none" stroke="{escape(color)}" stroke-width="{width}"'+(f' stroke-dasharray="{dash}"' if dash else '')+'/>')
    fontsize=max(11, round(min(w,h)*.035))
    def text(tx,ty,value,size=None,anchor='start',rotate=None):
        transform=f' transform="rotate({rotate} {tx} {ty})"' if rotate else ''
        parts.append(f'<text x="{tx}" y="{ty}" font-family="Arial, Microsoft YaHei, sans-serif" font-size="{size or fontsize}" text-anchor="{anchor}" fill="#111111"{transform}>{escape(value)}</text>')
    if settings.show_axes:
        if settings.show_grid:
            for v in np.linspace(0,hi,4)[1:]: line(left,py(v),right,py(v),'#d9d9d9')
        line(left,top,left,bottom); line(left,bottom,right,bottom)
        for v in np.linspace(-settings.x_half_range,settings.x_half_range,5):
            line(px(v),bottom,px(v),bottom+4)
            text(px(v),bottom+7+fontsize*.82,f'{v:g}',fontsize*.82,'middle')
        for v in np.linspace(0,hi,4):
            line(left-4,py(v),left,py(v))
            text(left-9,py(v)+fontsize*.28,f'{v:.1f}',fontsize*.82,'end')
        text((left+right)/2,h-8,settings.xlabel,anchor='middle')
        text(18,(top+bottom)/2,settings.ylabel,anchor='middle',rotate=-90)
    if settings.title.strip(): text(w/2,fontsize+6,settings.title,anchor='middle')
    if settings.show_centers:
        for center in (-settings.separation/2,settings.separation/2):
            line(px(center),top,px(center),bottom,'#b5b5b5',1,'2 3')
    if settings.show_baseline:
        line(left,py(float(baseline[0])),right,py(float(baseline[0])),'#7f7f7f',1,'3 4')
    dash=max(3,5.5*settings.component_width)
    gap=max(2,3.5*settings.component_width)
    for name,values,color,width,dashed in [
        ('gaussian-1',g1,settings.component1_color,settings.component_width,True),
        ('gaussian-2',g2,settings.component2_color,settings.component_width,True),
        ('sum',total,settings.sum_color,settings.sum_width,False),
    ]:
        coords=' '.join(('M' if i==0 else 'L')+f'{px(float(xv)):.5f},{py(float(yv)):.5f}' for i,(xv,yv) in enumerate(zip(x,values)))
        parts.append(f'<path id="{name}" d="{coords}" fill="none" stroke="{escape(color)}" stroke-width="{width}" stroke-linejoin="round"'+(f' stroke-dasharray="{dash} {gap}"' if dashed else '')+'/>')
    if settings.show_axes and settings.show_legend:
        for i,(label,color,dashed) in enumerate([('Gaussian 1',settings.component1_color,True),('Gaussian 2',settings.component2_color,True),('Sum',settings.sum_color,False)]):
            yy=top+fontsize+(fontsize+8)*i
            line(right-150,yy-4,right-115,yy-4,color,settings.component_width if dashed else settings.sum_width,'6 4' if dashed else '')
            text(right-105,yy,label,fontsize*.86)
    parts.append('</svg>')
    output.write_text('\n'.join(parts),encoding='utf-8')


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = ["DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
                  "arialbd.ttf" if bold else "arial.ttf"]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, max(8, size))
        except OSError:
            pass
    return ImageFont.load_default()


def _rgba(color: str, alpha: int = 255) -> tuple[int, int, int, int]:
    red, green, blue = ImageColor.getrgb(color)
    return red, green, blue, alpha


def _text_size(draw: ImageDraw.ImageDraw, text: str, font) -> tuple[int, int]:
    if not text:
        return 0, 0
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return right - left, bottom - top


def _dashed_polyline(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[float, float]],
    *,
    fill: tuple[int, int, int, int],
    width: int,
    dash: float,
    gap: float,
) -> None:
    """Draw a dashed polyline while keeping the dash phase continuous."""
    if len(points) < 2:
        return
    pattern = dash + gap
    distance_along = 0.0
    for start, end in zip(points[:-1], points[1:]):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.hypot(dx, dy)
        if length <= 0:
            continue
        segment_position = 0.0
        while segment_position < length:
            phase = (distance_along + segment_position) % pattern
            run = min(length - segment_position, pattern - phase)
            if phase < dash:
                run = min(run, dash - phase)
                t0 = segment_position / length
                t1 = (segment_position + run) / length
                p0 = (start[0] + dx * t0, start[1] + dy * t0)
                p1 = (start[0] + dx * t1, start[1] + dy * t1)
                draw.line((p0, p1), fill=fill, width=width)
            segment_position += max(run, 0.01)
        distance_along += length


def render_plot(
    settings: PlotSettings,
    width: int,
    height: int,
    *,
    transparent: bool,
    supersample: int = 3,
) -> Image.Image:
    """Render a high-quality RGBA plot using Pillow."""
    width = max(320, int(width))
    height = max(220, int(height))
    supersample = max(1, int(supersample))
    canvas_width = width * supersample
    canvas_height = height * supersample
    scale = supersample

    background = (255, 255, 255, 0) if transparent else (255, 255, 255, 255)
    image = Image.new("RGBA", (canvas_width, canvas_height), background)
    draw = ImageDraw.Draw(image)

    base_font_size = max(11, round(min(width, height) * 0.035)) * scale
    tick_font = _font(round(base_font_size * 0.82))
    label_font = _font(base_font_size)
    title_font = _font(round(base_font_size * 1.08), bold=True)
    legend_font = _font(round(base_font_size * 0.86))

    if settings.show_axes:
        left = 68 * scale
        right = canvas_width - 20 * scale
        top = (46 if settings.title.strip() else 22) * scale
        bottom = canvas_height - 58 * scale
    else:
        left = 8 * scale
        right = canvas_width - 8 * scale
        top = 8 * scale
        bottom = canvas_height - 8 * scale
    plot_width = max(1.0, right - left)
    plot_height = max(1.0, bottom - top)

    x, g1, g2, total, baseline = calculate_curves(settings)
    visible_max = max(float(np.max(g1)), float(np.max(g2)), float(np.max(total)))
    visible_min = min(0.0, float(np.min(total)), float(np.min(baseline)))
    y_span = max(visible_max - visible_min, 0.1)
    y_min = visible_min - 0.015 * y_span
    y_max = visible_max + 0.08 * y_span

    def px(value: float) -> float:
        return left + (value + settings.x_half_range) / (2 * settings.x_half_range) * plot_width

    def py(value: float) -> float:
        return bottom - (value - y_min) / (y_max - y_min) * plot_height

    foreground = _rgba("#111111")
    muted = _rgba("#707070")
    grid_color = _rgba("#D9D9D9", 180)
    guide_color = _rgba("#B5B5B5", 180)

    x_ticks = np.linspace(-settings.x_half_range, settings.x_half_range, 5)
    y_ticks = np.linspace(0.0, visible_max, 4)

    if settings.show_axes and settings.show_grid:
        for value in y_ticks[1:]:
            draw.line((left, py(float(value)), right, py(float(value))), fill=grid_color, width=scale)

    if settings.show_axes:
        axis_width = max(1, round(0.9 * scale))
        draw.line((left, top, left, bottom), fill=foreground, width=axis_width)
        draw.line((left, bottom, right, bottom), fill=foreground, width=axis_width)
        tick_length = 4 * scale

        for value in x_ticks:
            x_position = px(float(value))
            draw.line((x_position, bottom, x_position, bottom + tick_length), fill=foreground, width=scale)
            label = f"{value:g}"
            text_width, _ = _text_size(draw, label, tick_font)
            draw.text((x_position - text_width / 2, bottom + 7 * scale), label, font=tick_font, fill=muted)

        for value in y_ticks:
            y_position = py(float(value))
            draw.line((left - tick_length, y_position, left, y_position), fill=foreground, width=scale)
            label = f"{value:.1f}"
            text_width, text_height = _text_size(draw, label, tick_font)
            draw.text(
                (left - tick_length - text_width - 5 * scale, y_position - text_height / 2),
                label,
                font=tick_font,
                fill=muted,
            )

        xlabel_width, _ = _text_size(draw, settings.xlabel, label_font)
        draw.text(
            ((left + right - xlabel_width) / 2, canvas_height - 28 * scale),
            settings.xlabel,
            font=label_font,
            fill=foreground,
        )

        if settings.ylabel:
            text_width, text_height = _text_size(draw, settings.ylabel, label_font)
            label_image = Image.new("RGBA", (text_width + 8 * scale, text_height + 8 * scale), (0, 0, 0, 0))
            label_draw = ImageDraw.Draw(label_image)
            label_draw.text((4 * scale, 4 * scale), settings.ylabel, font=label_font, fill=foreground)
            rotated = label_image.rotate(90, expand=True, resample=RESAMPLING.BICUBIC)
            image.alpha_composite(
                rotated,
                (10 * scale, round((top + bottom - rotated.height) / 2)),
            )

        if settings.title.strip():
            title_width, _ = _text_size(draw, settings.title.strip(), title_font)
            draw.text(
                ((canvas_width - title_width) / 2, 7 * scale),
                settings.title.strip(),
                font=title_font,
                fill=foreground,
            )

    def series_points(values: np.ndarray) -> list[tuple[float, float]]:
        return [(px(float(xv)), py(float(yv))) for xv, yv in zip(x, values)]

    component_width = max(1, round(settings.component_width * scale))
    sum_width = max(1, round(settings.sum_width * scale))
    dash = max(3 * scale, 5.5 * settings.component_width * scale)
    gap = max(2 * scale, 3.5 * settings.component_width * scale)

    if settings.show_centers:
        center_peaks = (
            (-settings.separation / 2.0, float(np.max(g1))),
            (settings.separation / 2.0, float(np.max(g2))),
        )
        for center, peak in center_peaks:
            _dashed_polyline(
                draw,
                [(px(center), py(float(baseline[0]))), (px(center), py(float(baseline[0]) + peak))],
                fill=guide_color,
                width=max(1, scale),
                dash=2 * scale,
                gap=3 * scale,
            )

    if settings.show_baseline:
        _dashed_polyline(
            draw,
            [(left, py(float(baseline[0]))), (right, py(float(baseline[0])))],
            fill=_rgba("#7F7F7F"),
            width=max(1, scale),
            dash=3 * scale,
            gap=4 * scale,
        )

    _dashed_polyline(
        draw,
        series_points(g1),
        fill=_rgba(settings.component1_color),
        width=component_width,
        dash=dash,
        gap=gap,
    )
    _dashed_polyline(
        draw,
        series_points(g2),
        fill=_rgba(settings.component2_color),
        width=component_width,
        dash=dash,
        gap=gap,
    )
    draw.line(
        series_points(total),
        fill=_rgba(settings.sum_color),
        width=sum_width,
        joint="curve",
    )

    if settings.show_axes and settings.show_legend:
        legend_items = [
            ("Gaussian 1", settings.component1_color, True, component_width),
            ("Gaussian 2", settings.component2_color, True, component_width),
            ("Sum", settings.sum_color, False, sum_width),
        ]
        if settings.show_baseline:
            legend_items.append(("Background", "#7F7F7F", True, max(1, scale)))
        line_length = 28 * scale
        line_gap = 7 * scale
        row_height = max(18 * scale, round(base_font_size * 1.18))
        max_text_width = max(_text_size(draw, item[0], legend_font)[0] for item in legend_items)
        legend_width = line_length + line_gap + max_text_width
        legend_x = right - legend_width - 7 * scale
        legend_y = top + 8 * scale
        for index, (label, color, dashed, line_width) in enumerate(legend_items):
            y_position = legend_y + index * row_height + row_height / 2
            if dashed:
                _dashed_polyline(
                    draw,
                    [(legend_x, y_position), (legend_x + line_length, y_position)],
                    fill=_rgba(color),
                    width=max(1, line_width),
                    dash=6 * scale,
                    gap=4 * scale,
                )
            else:
                draw.line(
                    (legend_x, y_position, legend_x + line_length, y_position),
                    fill=_rgba(color),
                    width=max(1, line_width),
                )
            draw.text(
                (legend_x + line_length + line_gap, y_position - row_height * 0.36),
                label,
                font=legend_font,
                fill=foreground,
            )

    if supersample > 1:
        image = image.resize((width, height), RESAMPLING.LANCZOS)
    return image


class GaussianSignalApp(tk.Tk):
    """Tkinter editor for Gaussian components and their sum."""

    DEFAULTS = PlotSettings()

    def __init__(self) -> None:
        super().__init__()
        self.title("Gaussian Signal Overlay Designer")
        self.geometry("1320x850")
        self.minsize(1040, 670)

        self.vars: dict[str, tk.Variable] = {}
        self.value_labels: dict[str, ttk.Label] = {}
        self.color_buttons: dict[str, tk.Button] = {}
        self._preview_after: str | None = None
        self._preview_photo: ImageTk.PhotoImage | None = None
        self._ready = False

        self._configure_style()
        self._build_layout()
        self._ready = True
        self.after(100, self.refresh_preview)

    def _configure_style(self) -> None:
        style = ttk.Style(self)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("Title.TLabel", font=("Microsoft YaHei UI", 15, "bold"))
        style.configure("Hint.TLabel", foreground="#666666")

    def _build_layout(self) -> None:
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        left_shell = ttk.Frame(self, padding=(10, 10, 4, 10), width=380)
        left_shell.grid(row=0, column=0, sticky="nsew")
        left_shell.grid_propagate(False)
        left_shell.columnconfigure(0, weight=1)
        left_shell.rowconfigure(0, weight=1)

        scroll_canvas = tk.Canvas(left_shell, highlightthickness=0, width=355)
        scrollbar = ttk.Scrollbar(left_shell, orient="vertical", command=scroll_canvas.yview)
        scroll_canvas.configure(yscrollcommand=scrollbar.set)
        scroll_canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        controls = ttk.Frame(scroll_canvas, padding=(4, 0, 8, 8))
        controls_id = scroll_canvas.create_window((0, 0), window=controls, anchor="nw")
        controls.bind(
            "<Configure>",
            lambda _event: scroll_canvas.configure(scrollregion=scroll_canvas.bbox("all")),
        )
        scroll_canvas.bind(
            "<Configure>",
            lambda event: scroll_canvas.itemconfigure(controls_id, width=event.width),
        )
        scroll_canvas.bind_all(
            "<MouseWheel>",
            lambda event: scroll_canvas.yview_scroll(int(-event.delta / 120), "units"),
        )

        ttk.Label(controls, text="Gaussian Signal Designer", style="Title.TLabel").pack(
            anchor="w", pady=(2, 2)
        )
        ttk.Label(
            controls,
            text="Dashed lines show the two Gaussian components; the solid line shows their pointwise sum. The preview updates automatically.",
            style="Hint.TLabel",
            wraplength=330,
        ).pack(anchor="w", pady=(0, 10))

        gaussian_frame = self._section(controls, "Gaussian parameters")
        self._scale(gaussian_frame, "Peak separation d", "separation", 0.0, 7.0, 2.5, 2)
        self._scale(gaussian_frame, "Width σ₁", "sigma1", 0.20, 2.50, 1.0, 2)
        self._scale(gaussian_frame, "Width σ₂", "sigma2", 0.20, 2.50, 1.0, 2)
        self._scale(gaussian_frame, "Amplitude A₁", "amplitude1", 0.05, 2.00, 1.0, 2)
        self._scale(gaussian_frame, "Amplitude A₂", "amplitude2", 0.05, 2.00, 1.0, 2)
        self._scale(gaussian_frame, "Baseline B", "baseline", 0.0, 1.0, 0.0, 2)
        self._scale(gaussian_frame, "X-axis half-range", "x_half_range", 2.5, 12.0, 6.0, 1)
        self._check(gaussian_frame, "Normalize to sum peak", "normalize", False)

        style_frame = self._section(controls, "Lines and display")
        self._color(style_frame, "Component 1 color", "component1_color", "#7F7F7F")
        self._color(style_frame, "Component 2 color", "component2_color", "#A0A0A0")
        self._color(style_frame, "Sum color", "sum_color", "#111111")
        self._scale(style_frame, "Component line width", "component_width", 0.5, 4.0, 1.5, 1)
        self._scale(style_frame, "Sum line width", "sum_width", 0.8, 5.0, 2.4, 1)
        self._check(style_frame, "Show axes", "show_axes", True)
        self._check(style_frame, "Show legend", "show_legend", True)
        self._check(style_frame, "Show grid", "show_grid", False)
        self._check(style_frame, "Show peak centers", "show_centers", False)
        self._check(style_frame, "Show baseline", "show_baseline", False)

        labels_frame = self._section(controls, "Plot text")
        self._entry(labels_frame, "X-axis label", "xlabel", "Position")
        self._entry(labels_frame, "Y-axis label", "ylabel", "Signal")
        self._entry(labels_frame, "Plot title", "title", "")

        export_frame = self._section(controls, "PNG / SVG export")
        self._entry(export_frame, "Width (inches)", "figure_width", "5.6")
        self._entry(export_frame, "Height (inches)", "figure_height", "3.7")
        self._entry(export_frame, "Resolution (DPI)", "dpi", "300")
        ttk.Label(
            export_frame,
            text="PNG uses supersampling for smooth lines; SVG preserves editable vector paths. Hide the axes to export only the curves.",
            style="Hint.TLabel",
            wraplength=315,
        ).pack(anchor="w", pady=(5, 2))

        buttons = ttk.Frame(controls)
        buttons.pack(fill="x", pady=(12, 4))
        ttk.Button(buttons, text="Export transparent PNG…", command=self.export_png).pack(
            side="left", expand=True, fill="x", padx=(0, 4)
        )
        ttk.Button(buttons, text="Restore defaults", command=self.reset_defaults).pack(
            side="left", expand=True, fill="x", padx=(4, 0)
        )
        ttk.Button(controls, text="Export transparent SVG…", command=self.export_svg).pack(fill="x", pady=4)

        right = ttk.Frame(self, padding=(6, 10, 10, 10))
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)
        ttk.Label(right, text="Live preview", style="Title.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 6)
        )

        preview_border = ttk.Frame(right, relief="sunken", borderwidth=1)
        preview_border.grid(row=1, column=0, sticky="nsew")
        preview_border.columnconfigure(0, weight=1)
        preview_border.rowconfigure(0, weight=1)
        self.preview_label = ttk.Label(preview_border, anchor="center")
        self.preview_label.grid(row=0, column=0, sticky="nsew")
        preview_border.bind("<Configure>", self.schedule_preview)

        self.status = tk.StringVar(value="Ready")
        ttk.Label(right, textvariable=self.status, style="Hint.TLabel").grid(
            row=2, column=0, sticky="ew", pady=(5, 0)
        )

    @staticmethod
    def _section(parent: tk.Widget, title: str) -> ttk.LabelFrame:
        frame = ttk.LabelFrame(parent, text=title, padding=(9, 7))
        frame.pack(fill="x", pady=5)
        return frame

    def _scale(
        self,
        parent: tk.Widget,
        label: str,
        key: str,
        low: float,
        high: float,
        default: float,
        decimals: int,
    ) -> None:
        variable = tk.DoubleVar(value=default)
        self.vars[key] = variable
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=2)
        header = ttk.Frame(row)
        header.pack(fill="x")
        ttk.Label(header, text=label).pack(side="left")
        value_label = ttk.Label(header, width=8, anchor="e")
        value_label.pack(side="right")
        self.value_labels[key] = value_label
        ttk.Scale(
            row,
            from_=low,
            to=high,
            variable=variable,
            command=lambda _value, k=key, d=decimals: self._scale_changed(k, d),
        ).pack(fill="x", pady=(1, 0))
        self._update_value_label(key, decimals)

    def _entry(self, parent: tk.Widget, label: str, key: str, default: str) -> None:
        variable = tk.StringVar(value=default)
        self.vars[key] = variable
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=3)
        ttk.Label(row, text=label).pack(side="left")
        entry = ttk.Entry(row, textvariable=variable, width=18)
        entry.pack(side="right", fill="x", expand=True, padx=(10, 0))
        entry.bind("<KeyRelease>", self.schedule_preview)
        entry.bind("<FocusOut>", self.schedule_preview)

    def _check(self, parent: tk.Widget, label: str, key: str, default: bool) -> None:
        variable = tk.BooleanVar(value=default)
        self.vars[key] = variable
        ttk.Checkbutton(
            parent,
            text=label,
            variable=variable,
            command=self.schedule_preview,
        ).pack(anchor="w", pady=2)

    def _color(self, parent: tk.Widget, label: str, key: str, default: str) -> None:
        variable = tk.StringVar(value=default)
        self.vars[key] = variable
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=3)
        ttk.Label(row, text=label).pack(side="left")
        button = tk.Button(
            row,
            text=default.upper(),
            width=10,
            background=default,
            foreground=self._contrast_text(default),
            command=lambda k=key: self.choose_color(k),
        )
        button.pack(side="right")
        self.color_buttons[key] = button

    @staticmethod
    def _contrast_text(color: str) -> str:
        red, green, blue = ImageColor.getrgb(color)
        luminance = 0.299 * red + 0.587 * green + 0.114 * blue
        return "#000000" if luminance > 150 else "#FFFFFF"

    def choose_color(self, key: str) -> None:
        initial = str(self.vars[key].get())
        _rgb, chosen = colorchooser.askcolor(initialcolor=initial, parent=self)
        if not chosen:
            return
        chosen = chosen.upper()
        self.vars[key].set(chosen)
        self.color_buttons[key].configure(
            text=chosen,
            background=chosen,
            foreground=self._contrast_text(chosen),
        )
        self.schedule_preview()

    def _scale_changed(self, key: str, decimals: int) -> None:
        self._update_value_label(key, decimals)
        self.schedule_preview()

    def _update_value_label(self, key: str, decimals: int) -> None:
        self.value_labels[key].configure(text=f"{float(self.vars[key].get()):.{decimals}f}")

    def schedule_preview(self, _event=None) -> None:
        if not self._ready:
            return
        if self._preview_after is not None:
            self.after_cancel(self._preview_after)
        self._preview_after = self.after(90, self.refresh_preview)

    def get_settings(self) -> PlotSettings:
        def numeric_entry(key: str) -> float:
            return float(str(self.vars[key].get()).strip())

        settings = PlotSettings(
            separation=float(self.vars["separation"].get()),
            sigma1=float(self.vars["sigma1"].get()),
            sigma2=float(self.vars["sigma2"].get()),
            amplitude1=float(self.vars["amplitude1"].get()),
            amplitude2=float(self.vars["amplitude2"].get()),
            baseline=float(self.vars["baseline"].get()),
            x_half_range=float(self.vars["x_half_range"].get()),
            component_width=float(self.vars["component_width"].get()),
            sum_width=float(self.vars["sum_width"].get()),
            component1_color=str(self.vars["component1_color"].get()),
            component2_color=str(self.vars["component2_color"].get()),
            sum_color=str(self.vars["sum_color"].get()),
            show_axes=bool(self.vars["show_axes"].get()),
            show_legend=bool(self.vars["show_legend"].get()),
            show_grid=bool(self.vars["show_grid"].get()),
            show_centers=bool(self.vars["show_centers"].get()),
            show_baseline=bool(self.vars["show_baseline"].get()),
            normalize=bool(self.vars["normalize"].get()),
            xlabel=str(self.vars["xlabel"].get()),
            ylabel=str(self.vars["ylabel"].get()),
            title=str(self.vars["title"].get()),
            figure_width=numeric_entry("figure_width"),
            figure_height=numeric_entry("figure_height"),
            dpi=max(72, int(round(numeric_entry("dpi")))),
        )
        if settings.figure_width <= 0 or settings.figure_height <= 0:
            raise ValueError("Export width and height must be greater than zero.")
        if settings.dpi > 1200:
            raise ValueError("DPI must be between 72 and 1200.")
        return settings

    def refresh_preview(self) -> None:
        self._preview_after = None
        try:
            settings = self.get_settings()
        except (TypeError, ValueError) as exc:
            self.status.set(f"Invalid parameters: {exc}")
            return

        width = max(420, self.preview_label.winfo_width() - 8)
        height = max(300, self.preview_label.winfo_height() - 8)
        preview = render_plot(settings, width, height, transparent=False, supersample=2)
        self._preview_photo = ImageTk.PhotoImage(preview)
        self.preview_label.configure(image=self._preview_photo)

        ratio1 = settings.separation / settings.sigma1
        ratio2 = settings.separation / settings.sigma2
        self.status.set(
            f"d/σ₁ = {ratio1:.2f}, d/σ₂ = {ratio2:.2f}; "
            "equal-width, equal-amplitude Gaussians become bimodal when d/σ > 2."
        )

    def export_svg(self) -> None:
        try:
            settings = self.get_settings()
            output = filedialog.asksaveasfilename(parent=self, title="Export SVG vector image", initialdir=str(APP_DIR), initialfile="gaussian_signal_overlay.svg", defaultextension=".svg", filetypes=(("SVG vector image", "*.svg"),))
            if not output:
                return
            export_svg_file(settings, Path(output))
            self.status.set(f"SVG exported: {output}")
            messagebox.showinfo("Export complete", f"SVG vector image saved:\n{output}", parent=self)
        except Exception as exc:
            messagebox.showerror("Export failed", str(exc), parent=self)

    def export_png(self) -> None:
        try:
            settings = self.get_settings()
        except (TypeError, ValueError) as exc:
            messagebox.showerror("Invalid parameters", str(exc), parent=self)
            return

        output = filedialog.asksaveasfilename(
            parent=self,
            title="Export transparent PNG",
            initialdir=str(APP_DIR),
            initialfile="gaussian_signal_overlay.png",
            defaultextension=".png",
            filetypes=(("PNG image", "*.png"),),
        )
        if not output:
            return

        try:
            width = round(settings.figure_width * settings.dpi)
            height = round(settings.figure_height * settings.dpi)
            image = render_plot(settings, width, height, transparent=True, supersample=3)
            image.save(output, format="PNG", dpi=(settings.dpi, settings.dpi))
        except Exception as exc:  # pragma: no cover - GUI error path
            messagebox.showerror("Export failed", str(exc), parent=self)
            return

        self.status.set(f"Transparent PNG exported: {output}")
        messagebox.showinfo("Export complete", f"Transparent PNG saved:\n{output}", parent=self)

    def reset_defaults(self) -> None:
        defaults = self.DEFAULTS
        for key, variable in self.vars.items():
            if hasattr(defaults, key):
                variable.set(getattr(defaults, key))
        for key, button in self.color_buttons.items():
            color = str(self.vars[key].get()).upper()
            button.configure(
                text=color,
                background=color,
                foreground=self._contrast_text(color),
            )
        decimals = {
            "separation": 2,
            "sigma1": 2,
            "sigma2": 2,
            "amplitude1": 2,
            "amplitude2": 2,
            "baseline": 2,
            "x_half_range": 1,
            "component_width": 1,
            "sum_width": 1,
        }
        for key, places in decimals.items():
            self._update_value_label(key, places)
        self.schedule_preview()


def smoke_test(output: Path) -> None:
    """Export a default transparent PNG without opening the GUI."""
    output.parent.mkdir(parents=True, exist_ok=True)
    settings = replace(
        PlotSettings(),
        show_axes=False,
        show_legend=False,
        show_centers=True,
    )
    width = round(settings.figure_width * settings.dpi)
    height = round(settings.figure_height * settings.dpi)
    image = render_plot(settings, width, height, transparent=True, supersample=3)
    image.save(output, format="PNG", dpi=(settings.dpi, settings.dpi))
    print(output.resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="render a transparent PNG without opening the GUI",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=APP_DIR / "gaussian_signal_smoke_test.png",
        help="output path used with --smoke-test",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        smoke_test(args.output)
        return
    app = GaussianSignalApp()
    app.mainloop()


if __name__ == "__main__":
    main()
