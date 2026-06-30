from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scurve_diffusion_flow import ExperimentConfig, sample_s_curve, train_diffusion, train_flow_matching  # noqa: E402

ANALYSIS_PATH = ROOT / "scripts" / "generate-analysis-figures.py"
spec = importlib.util.spec_from_file_location("analysis_figures", ANALYSIS_PATH)
analysis = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(analysis)

Array = np.ndarray
SCALE = 3
BOUNDS = (-2.65, 2.65, -2.45, 2.45)

INK = analysis.INK
MUTED = analysis.MUTED
GRID = analysis.GRID
AXIS = analysis.AXIS
TARGET = analysis.TARGET
TARGET_STRONG = analysis.TARGET_STRONG
DIFFUSION = analysis.DIFFUSION
FLOW = analysis.FLOW
PAIRED_COLORS = analysis.PAIRED_COLORS


def font(size: int = 12, serif: bool = False) -> ImageFont.ImageFont:
    if serif:
        candidates = [
            Path("C:/Windows/Fonts/simsun.ttc"),
            Path("C:/Windows/Fonts/times.ttf"),
            Path("C:/Windows/Fonts/georgia.ttf"),
        ]
    else:
        candidates = [
            Path("C:/Windows/Fonts/msyh.ttc"),
            Path("C:/Windows/Fonts/simhei.ttf"),
            Path("C:/Windows/Fonts/segoeui.ttf"),
            Path("C:/Windows/Fonts/arial.ttf"),
        ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size * SCALE)
    return ImageFont.load_default()


def to_pixel(points: Array, rect: tuple[int, int, int, int]) -> Array:
    return analysis.to_pixel(points, BOUNDS, rect)


def draw_points(
    draw: ImageDraw.ImageDraw,
    points: Array,
    rect: tuple[int, int, int, int],
    color: tuple[int, int, int, int],
    radius: int = 2,
    limit: int | None = None,
) -> None:
    if limit is not None and len(points) > limit:
        points = points[:limit]
    pixels = to_pixel(points, rect)
    r = max(1, radius * SCALE)
    for x, y in pixels:
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color)


def blend(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return tuple(int(round(a[i] * (1 - t) + b[i] * t)) for i in range(3))


def arrow_head(
    draw: ImageDraw.ImageDraw,
    p0: tuple[float, float],
    p1: tuple[float, float],
    color: tuple[int, int, int, int],
    size: float = 9.0,
) -> None:
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    n = math.hypot(dx, dy)
    if n < 1e-6:
        return
    ux, uy = dx / n, dy / n
    px, py = -uy, ux
    tip = p1
    a = (p1[0] - size * ux + 0.48 * size * px, p1[1] - size * uy + 0.48 * size * py)
    b = (p1[0] - size * ux - 0.48 * size * px, p1[1] - size * uy - 0.48 * size * py)
    draw.polygon([tip, a, b], fill=color)


def dashed_line(
    draw: ImageDraw.ImageDraw,
    p0: tuple[float, float],
    p1: tuple[float, float],
    color: tuple[int, int, int, int],
    width: int,
    dash: float = 9.0,
) -> None:
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return
    steps = int(length // (dash * 2)) + 1
    ux, uy = dx / length, dy / length
    for i in range(steps):
        a = min(i * dash * 2, length)
        b = min(a + dash, length)
        draw.line((p0[0] + ux * a, p0[1] + uy * a, p0[0] + ux * b, p0[1] + uy * b), fill=color, width=width)


def equal_rect(left: int, top: int, width: int, height: int) -> tuple[int, int, int, int]:
    target_ratio = (BOUNDS[1] - BOUNDS[0]) / (BOUNDS[3] - BOUNDS[2])
    if width / height > target_ratio:
        plot_h = height
        plot_w = int(plot_h * target_ratio)
    else:
        plot_w = width
        plot_h = int(plot_w / target_ratio)
    x0 = left + (width - plot_w) // 2
    y0 = top + (height - plot_h) // 2
    return x0, y0, x0 + plot_w, y0 + plot_h


def draw_plot_base(draw: ImageDraw.ImageDraw, rect: tuple[int, int, int, int]) -> None:
    for frac in np.linspace(0.0, 1.0, 7):
        x = rect[0] + frac * (rect[2] - rect[0])
        y = rect[1] + frac * (rect[3] - rect[1])
        draw.line((x, rect[1], x, rect[3]), fill=GRID, width=1)
        draw.line((rect[0], y, rect[2], y), fill=GRID, width=1)
    zero = to_pixel(np.array([[0.0, 0.0]]), rect)[0]
    draw.line((zero[0], rect[1], zero[0], rect[3]), fill=AXIS, width=1)
    draw.line((rect[0], zero[1], rect[2], zero[1]), fill=AXIS, width=1)
    draw.rectangle(rect, outline=(88, 94, 104, 255), width=1)


def choose_path_indices(
    x0: Array,
    diffusion_hist: list[Array],
    flow_hist: list[Array],
    count: int = 9,
) -> Array:
    diffusion_points = np.stack(diffusion_hist, axis=0)
    flow_points = np.stack(flow_hist, axis=0)
    margin = 0.12
    inside_start = (
        (x0[:, 0] > BOUNDS[0] + 0.45)
        & (x0[:, 0] < BOUNDS[1] - 0.45)
        & (x0[:, 1] > BOUNDS[2] + 0.45)
        & (x0[:, 1] < BOUNDS[3] - 0.45)
    )
    inside_paths = (
        (diffusion_points[:, :, 0].min(axis=0) > BOUNDS[0] + margin)
        & (diffusion_points[:, :, 0].max(axis=0) < BOUNDS[1] - margin)
        & (diffusion_points[:, :, 1].min(axis=0) > BOUNDS[2] + margin)
        & (diffusion_points[:, :, 1].max(axis=0) < BOUNDS[3] - margin)
        & (flow_points[:, :, 0].min(axis=0) > BOUNDS[0] + margin)
        & (flow_points[:, :, 0].max(axis=0) < BOUNDS[1] - margin)
        & (flow_points[:, :, 1].min(axis=0) > BOUNDS[2] + margin)
        & (flow_points[:, :, 1].max(axis=0) < BOUNDS[3] - margin)
    )
    candidates = np.where(inside_start & inside_paths)[0]
    if len(candidates) < count:
        candidates = np.arange(len(x0))
    pts = x0[candidates]
    selected = [int(np.argmin(np.sum(pts * pts, axis=1)))]
    min_dist = np.sum((pts - pts[selected[0]]) ** 2, axis=1)
    while len(selected) < count:
        idx = int(np.argmax(min_dist))
        selected.append(idx)
        min_dist = np.minimum(min_dist, np.sum((pts - pts[idx]) ** 2, axis=1))
    return candidates[np.array(selected, dtype=int)]


def draw_path_panel(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    target: Array,
    hist: list[Array],
    idx: Array,
    title: str,
    subtitle: str,
) -> None:
    draw.text((rect[0], rect[1] - 54 * SCALE), title, fill=INK, font=font(18, serif=True))
    draw.text((rect[0], rect[1] - 25 * SCALE), subtitle, fill=MUTED, font=font(11))
    draw_plot_base(draw, rect)
    draw_points(draw, target, rect, TARGET, radius=1, limit=1100)
    for order, j in enumerate(idx):
        path = np.array([state[j] for state in hist], dtype=np.float64)
        pix = to_pixel(path, rect)
        start = tuple(pix[0])
        end = tuple(pix[-1])
        dashed_line(draw, start, end, (80, 88, 100, 65), width=2 * SCALE, dash=8 * SCALE)
        color = PAIRED_COLORS[order % len(PAIRED_COLORS)]
        base = (color[0], color[1], color[2])
        for k in range(len(pix) - 1):
            t = k / max(1, len(pix) - 2)
            rgb = blend((215, 219, 224), base, 0.45 + 0.55 * t)
            draw.line((pix[k, 0], pix[k, 1], pix[k + 1, 0], pix[k + 1, 1]), fill=(*rgb, 190), width=3 * SCALE)
        for k in [24, 52, 76]:
            if 0 < k < len(pix):
                arrow_head(draw, tuple(pix[k - 1]), tuple(pix[k]), (*base, 210), size=10 * SCALE)
        r = 6 * SCALE
        draw.ellipse((start[0] - r, start[1] - r, start[0] + r, start[1] + r), fill=(255, 255, 255, 230), outline=(35, 42, 52, 220), width=2 * SCALE)
        draw.ellipse((end[0] - r, end[1] - r, end[0] + r, end[1] + r), fill=(*base, 230), outline=(255, 255, 255, 160), width=1 * SCALE)


def trajectory_stats(hist: list[Array]) -> dict:
    pts = np.stack(hist, axis=0)
    deltas = np.diff(pts, axis=0)
    step_len = np.linalg.norm(deltas, axis=2)
    path_len = step_len.sum(axis=0)
    chord = np.linalg.norm(pts[-1] - pts[0], axis=1)
    valid = chord > 0.05
    ratio = path_len[valid] / chord[valid]
    d0 = deltas[:-1]
    d1 = deltas[1:]
    cos = np.sum(d0 * d1, axis=2) / ((np.linalg.norm(d0, axis=2) * np.linalg.norm(d1, axis=2)) + 1e-9)
    angles = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    cur = pts[:-1]
    radius = np.linalg.norm(cur, axis=2)
    radial = np.sum(deltas * cur, axis=2) / (radius + 1e-9)
    delta_norm = np.linalg.norm(deltas, axis=2)
    tangential = np.sqrt(np.maximum(delta_norm * delta_norm - radial * radial, 0.0))
    radial_mean = radial.mean(axis=1)
    tangential_mean = tangential.mean(axis=1)
    speed_mean = delta_norm.mean(axis=1)
    sign_switch = None
    for i in range(1, len(radial_mean)):
        if radial_mean[i - 1] < 0.0 <= radial_mean[i]:
            sign_switch = i
            break
    return {
        "turn_per_particle": angles.mean(axis=0),
        "radial_mean": radial_mean,
        "tangential_mean": tangential_mean,
        "speed_mean": speed_mean,
        "path_chord_ratio": ratio,
        "summary": {
            "mean_turn_angle": float(angles.mean()),
            "median_turn_angle": float(np.median(angles.mean(axis=0))),
            "p90_turn_angle": float(np.percentile(angles.mean(axis=0), 90)),
            "median_path_chord_ratio": float(np.median(ratio)),
            "mean_path_chord_ratio": float(np.mean(ratio)),
            "early_radial_mean_0_8": float(radial[:8].mean()),
            "early_tangential_mean_0_8": float(tangential[:8].mean()),
            "early_speed_mean_0_8": float(delta_norm[:8].mean()),
            "late_radial_mean_40_80": float(radial[40:80].mean()),
            "late_tangential_mean_40_80": float(tangential[40:80].mean()),
            "late_speed_mean_40_80": float(delta_norm[40:80].mean()),
            "radial_sign_switch_step": sign_switch,
        },
    }


def save_paired_paths(
    target: Array,
    x0: Array,
    diffusion_hist: list[Array],
    flow_hist: list[Array],
    out_path: Path,
) -> None:
    width, height = 1200 * SCALE, 540 * SCALE
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img, "RGBA")
    idx = choose_path_indices(x0, diffusion_hist, flow_hist, 9)
    rect_h = 450 * SCALE
    rect_w = int(rect_h * (BOUNDS[1] - BOUNDS[0]) / (BOUNDS[3] - BOUNDS[2]))
    left_rect = equal_rect(34 * SCALE, 78 * SCALE, rect_w + 54 * SCALE, rect_h)
    right_rect = equal_rect(682 * SCALE, 78 * SCALE, rect_w + 54 * SCALE, rect_h)
    draw_path_panel(draw, left_rect, target, diffusion_hist, idx, "Diffusion / DDIM", "局部修正带来更明显的折向")
    draw_path_panel(draw, right_rect, target, flow_hist, idx, "Flow Matching / ODE", "方向连续，先收缩再外扩")
    draw.rectangle((566 * SCALE, 170 * SCALE, 612 * SCALE, 173 * SCALE), fill=(80, 88, 100, 90))
    dashed_line(draw, (566 * SCALE, 208 * SCALE), (612 * SCALE, 208 * SCALE), (80, 88, 100, 115), width=2 * SCALE, dash=8 * SCALE)
    draw.text((625 * SCALE, 158 * SCALE), "实际路径", fill=INK, font=font(11))
    draw.text((625 * SCALE, 196 * SCALE), "端点弦线", fill=MUTED, font=font(11))
    draw.ellipse((578 * SCALE, 258 * SCALE, 596 * SCALE, 276 * SCALE), fill=(255, 255, 255, 230), outline=(35, 42, 52, 220), width=2 * SCALE)
    draw.text((625 * SCALE, 251 * SCALE), "同一高斯起点", fill=MUTED, font=font(11))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def draw_line_chart(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    series: list[tuple[str, Array, tuple[int, int, int, int], str]],
    y_min: float,
    y_max: float,
    x_max: int,
    sign_step: int | None = None,
) -> None:
    for frac in np.linspace(0.0, 1.0, 5):
        y = rect[3] - frac * (rect[3] - rect[1])
        draw.line((rect[0], y, rect[2], y), fill=GRID, width=1)
    for step in [0, 20, 40, 60, 80]:
        x = rect[0] + step / x_max * (rect[2] - rect[0])
        draw.line((x, rect[1], x, rect[3]), fill=(238, 240, 243, 255), width=1)
        draw.text((x - 7 * SCALE, rect[3] + 10 * SCALE), str(step), fill=MUTED, font=font(9))
    zero_y = rect[3] - (0.0 - y_min) / (y_max - y_min) * (rect[3] - rect[1])
    draw.line((rect[0], zero_y, rect[2], zero_y), fill=(108, 116, 128, 150), width=2)
    if sign_step is not None:
        x = rect[0] + sign_step / x_max * (rect[2] - rect[0])
        draw.line((x, rect[1], x, rect[3]), fill=(220, 127, 44, 170), width=3)
        draw.text((x + 8 * SCALE, rect[1] + 8 * SCALE), f"step {sign_step}", fill=(146, 84, 34, 230), font=font(10))
    draw.rectangle(rect, outline=(88, 94, 104, 255), width=1)
    for label, values, color, style in series:
        pts = []
        for i, value in enumerate(values):
            x = rect[0] + i / x_max * (rect[2] - rect[0])
            y = rect[3] - (value - y_min) / (y_max - y_min) * (rect[3] - rect[1])
            pts.append((x, y))
        for i, (a, b) in enumerate(zip(pts[:-1], pts[1:])):
            if style == "dash" and i % 4 in [2, 3]:
                continue
            if style == "dot" and i % 4 not in [0]:
                continue
            width = 4 * SCALE if style != "dot" else 3 * SCALE
            draw.line((a[0], a[1], b[0], b[1]), fill=color, width=width)


def save_radial_tangent_chart(diffusion_stats: dict, flow_stats: dict, out_path: Path) -> None:
    width, height = 1060 * SCALE, 560 * SCALE
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img, "RGBA")
    rect = (76 * SCALE, 62 * SCALE, width - 64 * SCALE, height - 72 * SCALE)
    all_values = np.concatenate(
        [
            diffusion_stats["radial_mean"],
            diffusion_stats["tangential_mean"],
            diffusion_stats["speed_mean"],
            flow_stats["radial_mean"],
            flow_stats["tangential_mean"],
            flow_stats["speed_mean"],
        ]
    )
    y_min = min(float(all_values.min()) * 1.20, -0.002)
    y_max = max(float(all_values.max()) * 1.18, 0.004)
    draw_line_chart(
        draw,
        rect,
        [
            ("Diff radial", diffusion_stats["radial_mean"], (47, 102, 179, 210), "solid"),
            ("Diff tangential", diffusion_stats["tangential_mean"], (47, 102, 179, 130), "dash"),
            ("Diff speed", diffusion_stats["speed_mean"], (31, 75, 145, 185), "dot"),
            ("Flow radial", flow_stats["radial_mean"], (46, 139, 95, 225), "solid"),
            ("Flow tangential", flow_stats["tangential_mean"], (46, 139, 95, 135), "dash"),
            ("Flow speed", flow_stats["speed_mean"], (24, 105, 66, 195), "dot"),
        ],
        y_min,
        y_max,
        x_max=79,
        sign_step=flow_stats["summary"]["radial_sign_switch_step"],
    )
    legend = [
        ("Diff radial", (47, 102, 179, 210), "solid"),
        ("Diff tang.", (47, 102, 179, 130), "dash"),
        ("Diff speed", (31, 75, 145, 185), "dot"),
        ("Flow radial", (46, 139, 95, 225), "solid"),
        ("Flow tang.", (46, 139, 95, 135), "dash"),
        ("Flow speed", (24, 105, 66, 195), "dot"),
    ]
    x0 = width - 304 * SCALE
    for i, (label, color, style) in enumerate(legend):
        y = 13 * SCALE + i * 22 * SCALE
        if style == "solid":
            draw.line((x0, y + 8 * SCALE, x0 + 32 * SCALE, y + 8 * SCALE), fill=color, width=4 * SCALE)
        elif style == "dash":
            dashed_line(draw, (x0, y + 8 * SCALE), (x0 + 32 * SCALE, y + 8 * SCALE), color, width=4 * SCALE, dash=8 * SCALE)
        else:
            for j in range(0, 32, 8):
                draw.ellipse((x0 + j * SCALE, y + 5 * SCALE, x0 + (j + 3) * SCALE, y + 8 * SCALE), fill=color)
        draw.text((x0 + 42 * SCALE, y), label, fill=INK, font=font(9))
    draw.text((rect[0] - 54 * SCALE, rect[1] - 2 * SCALE), "step Δx", fill=MUTED, font=font(9))
    draw.text((rect[2] - 16 * SCALE, rect[3] + 30 * SCALE), "step", fill=MUTED, font=font(10))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def save_turn_histogram(diffusion_stats: dict, flow_stats: dict, out_path: Path) -> None:
    d = diffusion_stats["turn_per_particle"]
    f = flow_stats["turn_per_particle"]
    width, height = 1060 * SCALE, 560 * SCALE
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img, "RGBA")
    rect = (78 * SCALE, 62 * SCALE, width - 60 * SCALE, height - 72 * SCALE)
    max_x = float(np.percentile(np.concatenate([d, f]), 99))
    max_x = max(max_x, 12.0)
    bins = np.linspace(0.0, max_x, 32)
    hd, _ = np.histogram(np.clip(d, 0.0, max_x), bins=bins, density=True)
    hf, _ = np.histogram(np.clip(f, 0.0, max_x), bins=bins, density=True)
    max_y = max(float(hd.max()), float(hf.max())) * 1.12
    for frac in np.linspace(0.0, 1.0, 5):
        y = rect[3] - frac * (rect[3] - rect[1])
        draw.line((rect[0], y, rect[2], y), fill=GRID, width=1)
    draw.rectangle(rect, outline=(88, 94, 104, 255), width=1)
    bar_w = (rect[2] - rect[0]) / (len(bins) - 1)
    for i, val in enumerate(hd):
        x = rect[0] + i * bar_w
        y = rect[3] - val / max_y * (rect[3] - rect[1])
        draw.rectangle((x, y, x + bar_w * 0.82, rect[3]), fill=(47, 102, 179, 94))
    for i, val in enumerate(hf):
        x = rect[0] + i * bar_w + bar_w * 0.17
        y = rect[3] - val / max_y * (rect[3] - rect[1])
        draw.rectangle((x, y, x + bar_w * 0.82, rect[3]), fill=(46, 139, 95, 112))
    for value, color, label, offset in [
        (diffusion_stats["summary"]["mean_turn_angle"], (47, 102, 179, 230), "Diff mean", 0),
        (flow_stats["summary"]["mean_turn_angle"], (46, 139, 95, 230), "Flow mean", 1),
    ]:
        x = rect[0] + value / max_x * (rect[2] - rect[0])
        draw.line((x, rect[1], x, rect[3]), fill=color, width=3 * SCALE)
        draw.text((x + 7 * SCALE, rect[1] + (8 + offset * 25) * SCALE), f"{label} {value:.2f}°", fill=color, font=font(10))
    draw.text((rect[0], rect[3] + 11 * SCALE), "0°", fill=MUTED, font=font(9))
    draw.text((rect[2] - 34 * SCALE, rect[3] + 11 * SCALE), f"{max_x:.0f}°", fill=MUTED, font=font(9))
    draw.rectangle((width - 218 * SCALE, 22 * SCALE, width - 202 * SCALE, 38 * SCALE), fill=(47, 102, 179, 150))
    draw.text((width - 194 * SCALE, 19 * SCALE), "Diffusion", fill=INK, font=font(10))
    draw.rectangle((width - 218 * SCALE, 51 * SCALE, width - 202 * SCALE, 67 * SCALE), fill=(46, 139, 95, 160))
    draw.text((width - 194 * SCALE, 48 * SCALE), "Flow", fill=INK, font=font(10))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def main() -> None:
    out_dir = ROOT / "outputs" / "analysis_figures"
    asset_dir = ROOT / "presentation_v2" / "assets"
    out_dir.mkdir(parents=True, exist_ok=True)
    asset_dir.mkdir(parents=True, exist_ok=True)

    cfg = ExperimentConfig(seed=7, train_steps=2200, batch_size=512, sample_count=2500, sample_steps=80, frames=28)
    master_rng = np.random.default_rng(cfg.seed)
    diffusion_rng = np.random.default_rng(master_rng.integers(0, 2**32 - 1))
    flow_rng = np.random.default_rng(master_rng.integers(0, 2**32 - 1))
    eval_rng = np.random.default_rng(master_rng.integers(0, 2**32 - 1))

    print("training diffusion model for transport geometry")
    diffusion_model, _ = train_diffusion(cfg, diffusion_rng)
    print("training flow matching model for transport geometry")
    flow_model, _ = train_flow_matching(cfg, flow_rng)

    target = sample_s_curve(cfg.sample_count, eval_rng)
    x0, diffusion_hist, flow_hist, _ = analysis.sample_with_history(
        diffusion_model,
        flow_model,
        cfg,
        eval_rng,
        cfg.sample_count,
    )
    diffusion_stats = trajectory_stats(diffusion_hist)
    flow_stats = trajectory_stats(flow_hist)

    figures = {
        "10_paired_paths.png": lambda p: save_paired_paths(target, x0, diffusion_hist, flow_hist, p),
        "10_radial_tangent.png": lambda p: save_radial_tangent_chart(diffusion_stats, flow_stats, p),
        "10_turn_histogram.png": lambda p: save_turn_histogram(diffusion_stats, flow_stats, p),
    }
    for name, fn in figures.items():
        out_path = out_dir / name
        fn(out_path)
        (asset_dir / name).write_bytes(out_path.read_bytes())

    metrics = {
        "diffusion": diffusion_stats["summary"],
        "flow_matching": flow_stats["summary"],
        "notes": {
            "turn_angle": "mean angle between consecutive per-particle displacement vectors",
            "path_chord_ratio": "sum of step lengths divided by final displacement length; particles with chord <= 0.05 are excluded",
            "radial_projection": "signed projection of each step onto the current position vector; negative means inward motion",
            "speed_mean": "mean per-step displacement magnitude; proportional to speed when step size is fixed",
        },
    }
    metrics_path = out_dir / "10_transport_geometry_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"saved transport geometry figures to {out_dir} and {asset_dir}")


if __name__ == "__main__":
    main()
