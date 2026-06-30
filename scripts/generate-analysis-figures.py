from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scurve_diffusion_flow import (  # noqa: E402
    ExperimentConfig,
    MLP,
    diffusion_alpha_sigma,
    sample_s_curve,
    sliced_wasserstein_distance,
    time_features,
    to_pixel,
    train_diffusion,
    train_flow_matching,
)

Array = np.ndarray

SCALE = 2
BOUNDS = (-2.65, 2.65, -2.45, 2.45)
INK = (27, 33, 44, 255)
MUTED = (86, 95, 110, 255)
GRID = (224, 228, 232, 255)
AXIS = (178, 185, 196, 255)
TARGET = (235, 129, 37, 115)
TARGET_STRONG = (235, 129, 37, 210)
DIFFUSION = (31, 104, 196, 200)
FLOW = (30, 142, 82, 210)
PAIRED_COLORS = [
    (32, 92, 170, 220),
    (207, 92, 54, 220),
    (35, 130, 118, 220),
    (148, 84, 163, 220),
    (218, 150, 42, 220),
    (89, 112, 54, 220),
    (188, 75, 126, 220),
    (74, 119, 170, 220),
    (166, 103, 48, 220),
    (64, 142, 76, 220),
    (92, 86, 170, 220),
    (170, 68, 74, 220),
    (58, 132, 154, 220),
    (128, 128, 42, 220),
]


def font(size: int = 12) -> ImageFont.ImageFont:
    font_candidates = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/segoeui.ttf"),
    ]
    for candidate in font_candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size * SCALE)
    return ImageFont.load_default()


def save_scaled(img: Image.Image, out_path: Path) -> None:
    """Render at high resolution, then keep the high-res image for crisp deck display."""
    img.save(out_path)


def draw_plot_base(
    draw: ImageDraw.ImageDraw,
    width: int,
    height: int,
    title: str,
    subtitle: str | None = None,
    rect: tuple[int, int, int, int] | None = None,
) -> tuple[int, int, int, int]:
    rect = rect or (74, 78, width - 38, height - 54)
    f = font()
    title_x = max(24, rect[0] - 50)
    draw.text((title_x, 20), title, fill=INK, font=f)
    if subtitle:
        draw.text((title_x, 40), subtitle, fill=MUTED, font=f)
    for frac in np.linspace(0.0, 1.0, 7):
        x = rect[0] + frac * (rect[2] - rect[0])
        y = rect[1] + frac * (rect[3] - rect[1])
        draw.line((x, rect[1], x, rect[3]), fill=GRID)
        draw.line((rect[0], y, rect[2], y), fill=GRID)
    zero = to_pixel(np.array([[0.0, 0.0]]), BOUNDS, rect)[0]
    draw.line((zero[0], rect[1], zero[0], rect[3]), fill=AXIS)
    draw.line((rect[0], zero[1], rect[2], zero[1]), fill=AXIS)
    draw.rectangle(rect, outline=(88, 94, 104, 255), width=1)
    return rect


def draw_points(
    draw: ImageDraw.ImageDraw,
    points: Array,
    rect: tuple[int, int, int, int],
    color: tuple[int, int, int, int],
    radius: int = 1,
    limit: int | None = None,
) -> None:
    if limit is not None and len(points) > limit:
        points = points[:limit]
    pixels = to_pixel(points, BOUNDS, rect)
    radius = max(1, radius * SCALE)
    for x, y in pixels:
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)


def blend(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return tuple(int(round(a[i] * (1.0 - t) + b[i] * t)) for i in range(3))


def arrow_head(
    draw: ImageDraw.ImageDraw,
    p0: tuple[float, float],
    p1: tuple[float, float],
    color: tuple[int, int, int, int],
    size: float = 6.0,
) -> None:
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    n = math.hypot(dx, dy)
    if n < 1e-6:
        return
    ux, uy = dx / n, dy / n
    px, py = -uy, ux
    a = (p1[0] - size * ux + 0.45 * size * px, p1[1] - size * uy + 0.45 * size * py)
    b = (p1[0] - size * ux - 0.45 * size * px, p1[1] - size * uy - 0.45 * size * py)
    draw.polygon([p1, a, b], fill=color)


def predict_diffusion_delta(model: MLP, points: Array, t_cur: float, t_next: float) -> Array:
    t_batch = np.full((len(points), 1), t_cur, dtype=np.float64)
    eps_pred = model.predict(time_features(points, t_batch))
    alpha, sigma = diffusion_alpha_sigma(t_cur)
    alpha_next, sigma_next = diffusion_alpha_sigma(t_next)
    x0_hat = (points - sigma * eps_pred) / max(float(alpha), 1e-3)
    x0_hat = np.clip(x0_hat, -4.0, 4.0)
    next_points = alpha_next * x0_hat + sigma_next * eps_pred
    return next_points - points


def predict_flow_delta(model: MLP, points: Array, t_cur: float, dt: float) -> Array:
    t_batch = np.full((len(points), 1), t_cur, dtype=np.float64)
    return dt * model.predict(time_features(points, t_batch))


def sample_with_history(
    diffusion_model: MLP,
    flow_model: MLP,
    cfg: ExperimentConfig,
    rng: np.random.Generator,
    n: int,
) -> tuple[Array, list[Array], list[Array], Array]:
    x0 = rng.normal(size=(n, 2))
    diffusion = x0.copy()
    flow = x0.copy()
    diffusion_hist = [diffusion.copy()]
    flow_hist = [flow.copy()]

    diffusion_times = np.linspace(cfg.diffusion_t_max, 0.0, cfg.sample_steps + 1)
    flow_times = np.linspace(0.0, 1.0, cfg.sample_steps + 1)

    for i in range(cfg.sample_steps):
        diffusion += predict_diffusion_delta(diffusion_model, diffusion, float(diffusion_times[i]), float(diffusion_times[i + 1]))
        flow += predict_flow_delta(flow_model, flow, float(flow_times[i]), float(flow_times[i + 1] - flow_times[i]))
        diffusion_hist.append(diffusion.copy())
        flow_hist.append(flow.copy())
    return x0, diffusion_hist, flow_hist, flow_times


def save_vector_field_panel(
    diffusion_model: MLP,
    flow_model: MLP,
    target: Array,
    out_path: Path,
) -> None:
    width, height = 1280 * SCALE, 560 * SCALE
    panel_w = width // 2
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img, "RGBA")
    titles = [
        ("Diffusion one-step update field", "arrows show one deterministic reverse step around t=0.55"),
        ("Flow matching velocity field", "arrows show Euler ODE direction around t=0.45"),
    ]
    xs = np.linspace(-2.15, 2.15, 17)
    ys = np.linspace(-1.95, 1.95, 15)
    grid = np.array([(x, y) for y in ys for x in xs], dtype=np.float64)

    for panel, (title, subtitle) in enumerate(titles):
        left = panel * panel_w
        rect = draw_plot_base(draw, panel_w, height, title, subtitle, rect=(74 + left, 78, left + panel_w - 34, height - 54))
        draw_points(draw, target, rect, TARGET, radius=1, limit=1200)
        if panel == 0:
            delta = predict_diffusion_delta(diffusion_model, grid, 0.55, 0.55 - 0.98 / 80)
            color = DIFFUSION
        else:
            delta = predict_flow_delta(flow_model, grid, 0.45, 1.0 / 80)
            color = FLOW
        norms = np.linalg.norm(delta, axis=1)
        scale = 0.23 / max(float(np.percentile(norms, 90)), 1e-6)
        starts = to_pixel(grid, BOUNDS, rect)
        ends = to_pixel(grid + delta * scale, BOUNDS, rect)
        for p0, p1, n in zip(starts, ends, norms):
            if n < 1e-5:
                continue
            draw.line((p0[0], p0[1], p1[0], p1[1]), fill=color, width=4)
            arrow_head(draw, (p0[0], p0[1]), (p1[0], p1[1]), color, size=11.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_scaled(img, out_path)


def save_trajectory_panel(
    target: Array,
    diffusion_hist: list[Array],
    flow_hist: list[Array],
    out_path: Path,
    count: int = 12,
) -> None:
    width, height = 1280 * SCALE, 560 * SCALE
    panel_w = width // 2
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img, "RGBA")
    idx = choose_representative_indices(diffusion_hist[0], count)

    for panel, (title, subtitle, hist, color) in enumerate(
        [
            ("Diffusion particle paths", "same color = same initial particle; open circle = shared start", diffusion_hist, DIFFUSION),
            ("Flow matching particle paths", "paired with the left panel by color and starting point", flow_hist, FLOW),
        ]
    ):
        left = panel * panel_w
        rect = draw_plot_base(draw, panel_w, height, title, subtitle, rect=(74 + left, 78, left + panel_w - 34, height - 54))
        draw_points(draw, target, rect, TARGET, radius=1, limit=1100)
        for order, j in enumerate(idx):
            path_color = PAIRED_COLORS[order % len(PAIRED_COLORS)]
            path = np.array([state[j] for state in hist], dtype=np.float64)
            pix = to_pixel(path, BOUNDS, rect)
            for k in range(len(pix) - 1):
                t = k / max(1, len(pix) - 2)
                base = (path_color[0], path_color[1], path_color[2])
                rgb = blend((214, 218, 224), base, 0.35 + 0.65 * t)
                draw.line((pix[k, 0], pix[k, 1], pix[k + 1, 0], pix[k + 1, 1]), fill=(*rgb, 190), width=2)
            sx, sy = pix[0]
            ex, ey = pix[-1]
            r = 7
            draw.ellipse((sx - r, sy - r, sx + r, sy + r), fill=(255, 255, 255, 220), outline=(35, 42, 52, 210), width=3)
            draw.ellipse((ex - r, ey - r, ex + r, ey + r), fill=path_color)
            draw.line((ex - 8, ey, ex + 8, ey), fill=(255, 255, 255, 170), width=2)
            draw.line((ex, ey - 8, ex, ey + 8), fill=(255, 255, 255, 170), width=2)
    save_scaled(img, out_path)


def choose_representative_indices(points: Array, count: int) -> Array:
    in_bounds = np.where(
        (points[:, 0] > BOUNDS[0] + 0.25)
        & (points[:, 0] < BOUNDS[1] - 0.25)
        & (points[:, 1] > BOUNDS[2] + 0.25)
        & (points[:, 1] < BOUNDS[3] - 0.25)
    )[0]
    candidates = in_bounds if len(in_bounds) >= count else np.arange(len(points))
    candidate_points = points[candidates]
    center_idx = int(np.argmin(np.sum(candidate_points * candidate_points, axis=1)))
    selected = [center_idx]
    min_dist = np.sum((candidate_points - candidate_points[center_idx]) ** 2, axis=1)
    while len(selected) < count:
        next_idx = int(np.argmax(min_dist))
        selected.append(next_idx)
        dist = np.sum((candidate_points - candidate_points[next_idx]) ** 2, axis=1)
        min_dist = np.minimum(min_dist, dist)
    return candidates[np.array(selected, dtype=int)]


def save_snapshot_strip(
    target: Array,
    diffusion_hist: list[Array],
    flow_hist: list[Array],
    out_path: Path,
) -> None:
    cols = [0, 8, 20, 40, 80]
    width, height = 1500 * SCALE, 620 * SCALE
    cell_w = width // len(cols)
    row_h = height // 2
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img, "RGBA")
    f = font()
    labels = ["step 0", "step 8", "step 20", "step 40", "step 80"]
    rows = [("Diffusion / DDIM", diffusion_hist, DIFFUSION), ("Flow matching / ODE", flow_hist, FLOW)]
    for r, (row_name, hist, color) in enumerate(rows):
        draw.text((14, r * row_h + 16), row_name, fill=INK, font=f)
        for c, step in enumerate(cols):
            left = c * cell_w
            top = r * row_h
            rect = (left + 42, top + 54, left + cell_w - 22, top + row_h - 36)
            for frac in np.linspace(0.0, 1.0, 5):
                x = rect[0] + frac * (rect[2] - rect[0])
                y = rect[1] + frac * (rect[3] - rect[1])
                draw.line((x, rect[1], x, rect[3]), fill=(232, 235, 238, 255))
                draw.line((rect[0], y, rect[2], y), fill=(232, 235, 238, 255))
            draw.rectangle(rect, outline=(122, 128, 138, 255), width=1)
            draw.text((left + 42, top + 34), labels[c], fill=MUTED, font=f)
            draw_points(draw, target, rect, TARGET, radius=1, limit=900)
            draw_points(draw, hist[step], rect, color, radius=1, limit=900)
    save_scaled(img, out_path)


def nearest_distances(points: Array, target: Array, chunk: int = 512) -> Array:
    out = []
    for i in range(0, len(points), chunk):
        p = points[i : i + chunk]
        d2 = np.sum((p[:, None, :] - target[None, :, :]) ** 2, axis=2)
        out.append(np.sqrt(np.min(d2, axis=1)))
    return np.concatenate(out)


def save_distance_histogram(target: Array, diffusion: Array, flow: Array, out_path: Path) -> dict:
    d_diff = nearest_distances(diffusion, target)
    d_flow = nearest_distances(flow, target)
    width, height = 920 * SCALE, 520 * SCALE
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img, "RGBA")
    rect = (76, 74, width - 40, height - 70)
    draw.text((24, 22), "Distance to target manifold", fill=INK, font=font())
    draw.text((24, 42), "nearest-neighbor distance to target samples; lower means tighter curve fit", fill=MUTED, font=font())
    max_x = float(np.percentile(np.concatenate([d_diff, d_flow]), 99))
    bins = np.linspace(0.0, max_x, 34)
    h_diff, _ = np.histogram(d_diff, bins=bins, density=True)
    h_flow, _ = np.histogram(d_flow, bins=bins, density=True)
    max_y = max(float(h_diff.max()), float(h_flow.max())) * 1.08
    for frac in np.linspace(0.0, 1.0, 6):
        y = rect[3] - frac * (rect[3] - rect[1])
        draw.line((rect[0], y, rect[2], y), fill=GRID)
    draw.rectangle(rect, outline=(86, 94, 104, 255), width=1)
    bar_w = (rect[2] - rect[0]) / (len(bins) - 1)
    for i, val in enumerate(h_diff):
        x0 = rect[0] + i * bar_w
        x1 = x0 + bar_w * 0.85
        y = rect[3] - val / max_y * (rect[3] - rect[1])
        draw.rectangle((x0, y, x1, rect[3]), fill=(31, 104, 196, 95))
    for i, val in enumerate(h_flow):
        x0 = rect[0] + i * bar_w + bar_w * 0.15
        x1 = x0 + bar_w * 0.85
        y = rect[3] - val / max_y * (rect[3] - rect[1])
        draw.rectangle((x0, y, x1, rect[3]), fill=(30, 142, 82, 105))
    draw.text((rect[0], rect[3] + 16), "0", fill=MUTED, font=font())
    draw.text((rect[2] - 70, rect[3] + 16), f"{max_x:.2f}", fill=MUTED, font=font())
    draw.rectangle((width - 250, 86, width - 236, 100), fill=DIFFUSION)
    draw.text((width - 228, 84), f"Diffusion median {np.median(d_diff):.3f}", fill=INK, font=font())
    draw.rectangle((width - 250, 112, width - 236, 126), fill=FLOW)
    draw.text((width - 228, 110), f"Flow median {np.median(d_flow):.3f}", fill=INK, font=font())
    save_scaled(img, out_path)
    return {
        "diffusion_median_target_distance": float(np.median(d_diff)),
        "flow_median_target_distance": float(np.median(d_flow)),
        "diffusion_p90_target_distance": float(np.percentile(d_diff, 90)),
        "flow_p90_target_distance": float(np.percentile(d_flow, 90)),
    }


def curve_metrics(hist: list[Array]) -> tuple[Array, Array]:
    points = np.stack(hist, axis=0)
    deltas = np.diff(points, axis=0)
    step = np.linalg.norm(deltas, axis=2).mean(axis=1)
    d0 = deltas[:-1]
    d1 = deltas[1:]
    cos = np.sum(d0 * d1, axis=2) / ((np.linalg.norm(d0, axis=2) * np.linalg.norm(d1, axis=2)) + 1e-9)
    angle = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))).mean(axis=1)
    return step, angle


def smooth(values: Array, window: int = 5) -> Array:
    if len(values) < window:
        return values
    kernel = np.ones(window, dtype=np.float64) / window
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(values, (pad_left, pad_right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def clip_plot_points(points: list[tuple[float, float]], rect: tuple[int, int, int, int]) -> list[tuple[float, float]]:
    return [(x, float(np.clip(y, rect[1], rect[3]))) for x, y in points]


def save_step_curves(
    target: Array,
    diffusion_hist: list[Array],
    flow_hist: list[Array],
    out_path: Path,
) -> dict:
    diff_step, diff_angle = curve_metrics(diffusion_hist)
    flow_step, flow_angle = curve_metrics(flow_hist)
    rng = np.random.default_rng(12345)
    swd_diff = []
    swd_flow = []
    for i in range(len(diffusion_hist)):
        swd_diff.append(sliced_wasserstein_distance(diffusion_hist[i], target, rng, projections=64))
        swd_flow.append(sliced_wasserstein_distance(flow_hist[i], target, rng, projections=64))

    width, height = 1280 * SCALE, 620 * SCALE
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img, "RGBA")
    draw.text((24, 20), "Step-wise comparison curves", fill=INK, font=font())
    draw.text((24, 40), "distance drops, step length and turn angle describe the generated dynamics", fill=MUTED, font=font())
    panels = [
        ("Sliced Wasserstein to target", smooth(np.array(swd_diff)), smooth(np.array(swd_flow)), "lower is closer"),
        ("Mean step length", smooth(diff_step), smooth(flow_step), "smoothed transport magnitude"),
        ("Mean turn angle", smooth(diff_angle), smooth(flow_angle), "smoothed direction changes"),
    ]
    panel_w = (width - 80) // 3
    for p, (title, y_diff, y_flow, subtitle) in enumerate(panels):
        left = 40 + p * panel_w
        rect = (left + 46, 92, left + panel_w - 26, height - 70)
        draw.text((left + 8, 70), title, fill=INK, font=font())
        draw.text((left + 8, height - 44), subtitle, fill=MUTED, font=font())
        all_y = np.concatenate([y_diff, y_flow])
        ymin = 0.0
        ymax = float(np.max(all_y) * 1.08 + 1e-9)
        for frac in np.linspace(0.0, 1.0, 5):
            y = rect[3] - frac * (rect[3] - rect[1])
            draw.line((rect[0], y, rect[2], y), fill=GRID)
        draw.rectangle(rect, outline=(86, 94, 104, 255), width=1)

        def pts(values: Array) -> list[tuple[float, float]]:
            xs = np.linspace(rect[0], rect[2], len(values))
            ys = rect[3] - (values - ymin) / (ymax - ymin) * (rect[3] - rect[1])
            return clip_plot_points(list(zip(xs, ys)), rect)

        for color, vals in [(DIFFUSION, y_diff), (FLOW, y_flow)]:
            points = pts(vals)
            for a, b in zip(points[:-1], points[1:]):
                draw.line((a[0], a[1], b[0], b[1]), fill=color, width=3)
        draw.text((rect[0], rect[3] + 10), "0", fill=MUTED, font=font())
        draw.text((rect[2] - 24, rect[3] + 10), "80", fill=MUTED, font=font())
    draw.rectangle((width - 246, 20, width - 232, 34), fill=DIFFUSION)
    draw.text((width - 224, 18), "Diffusion", fill=INK, font=font())
    draw.rectangle((width - 140, 20, width - 126, 34), fill=FLOW)
    draw.text((width - 118, 18), "Flow", fill=INK, font=font())
    save_scaled(img, out_path)
    return {
        "diffusion_mean_turn_angle": float(np.mean(diff_angle)),
        "flow_mean_turn_angle": float(np.mean(flow_angle)),
        "diffusion_mean_step_length": float(np.mean(diff_step)),
        "flow_mean_step_length": float(np.mean(flow_step)),
    }


def per_particle_turn_angle(hist: list[Array]) -> Array:
    points = np.stack(hist, axis=0)
    deltas = np.diff(points, axis=0)
    d0 = deltas[:-1]
    d1 = deltas[1:]
    cos = np.sum(d0 * d1, axis=2) / ((np.linalg.norm(d0, axis=2) * np.linalg.norm(d1, axis=2)) + 1e-9)
    angles = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    return np.mean(angles, axis=0)


def save_smoothness_histogram(diffusion_hist: list[Array], flow_hist: list[Array], out_path: Path) -> dict:
    d = per_particle_turn_angle(diffusion_hist)
    f = per_particle_turn_angle(flow_hist)
    width, height = 920 * SCALE, 520 * SCALE
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img, "RGBA")
    rect = (76, 78, width - 40, height - 70)
    draw.text((24, 22), "Path smoothness distribution", fill=INK, font=font())
    draw.text((24, 42), "mean turn angle per particle; lower means a smoother generated path", fill=MUTED, font=font())

    max_x = float(np.percentile(np.concatenate([d, f]), 98))
    bins = np.linspace(0.0, max_x, 36)
    h_d, _ = np.histogram(np.clip(d, 0.0, max_x), bins=bins, density=True)
    h_f, _ = np.histogram(np.clip(f, 0.0, max_x), bins=bins, density=True)
    max_y = max(float(h_d.max()), float(h_f.max())) * 1.08
    for frac in np.linspace(0.0, 1.0, 6):
        y = rect[3] - frac * (rect[3] - rect[1])
        draw.line((rect[0], y, rect[2], y), fill=GRID)
    draw.rectangle(rect, outline=(86, 94, 104, 255), width=1)
    bar_w = (rect[2] - rect[0]) / (len(bins) - 1)
    for i, val in enumerate(h_d):
        x0 = rect[0] + i * bar_w
        x1 = x0 + bar_w * 0.85
        y = rect[3] - val / max_y * (rect[3] - rect[1])
        draw.rectangle((x0, y, x1, rect[3]), fill=(31, 104, 196, 95))
    for i, val in enumerate(h_f):
        x0 = rect[0] + i * bar_w + bar_w * 0.15
        x1 = x0 + bar_w * 0.85
        y = rect[3] - val / max_y * (rect[3] - rect[1])
        draw.rectangle((x0, y, x1, rect[3]), fill=(30, 142, 82, 105))
    draw.text((rect[0], rect[3] + 16), "0 deg", fill=MUTED, font=font())
    draw.text((rect[2] - 72, rect[3] + 16), f"{max_x:.1f} deg", fill=MUTED, font=font())
    draw.rectangle((width - 286, 86, width - 272, 100), fill=DIFFUSION)
    draw.text((width - 264, 84), f"Diffusion median {np.median(d):.2f} deg", fill=INK, font=font())
    draw.rectangle((width - 286, 112, width - 272, 126), fill=FLOW)
    draw.text((width - 264, 110), f"Flow median {np.median(f):.2f} deg", fill=INK, font=font())
    save_scaled(img, out_path)
    return {
        "diffusion_median_turn_angle": float(np.median(d)),
        "flow_median_turn_angle": float(np.median(f)),
        "diffusion_p90_turn_angle": float(np.percentile(d, 90)),
        "flow_p90_turn_angle": float(np.percentile(f, 90)),
    }


def sample_diffusion_from_initial(model: MLP, cfg: ExperimentConfig, x_init: Array, steps: int) -> Array:
    x = x_init.copy()
    times = np.linspace(cfg.diffusion_t_max, 0.0, steps + 1)
    for i in range(steps):
        x += predict_diffusion_delta(model, x, float(times[i]), float(times[i + 1]))
    return x


def sample_flow_from_initial(model: MLP, x_init: Array, steps: int) -> Array:
    x = x_init.copy()
    times = np.linspace(0.0, 1.0, steps + 1)
    for i in range(steps):
        x += predict_flow_delta(model, x, float(times[i]), float(times[i + 1] - times[i]))
    return x


def save_nfe_sensitivity(
    diffusion_model: MLP,
    flow_model: MLP,
    target: Array,
    x_init: Array,
    out_path: Path,
) -> dict:
    nfes = [10, 20, 40, 80]
    diff_values = []
    flow_values = []
    for nfe in nfes:
        rng = np.random.default_rng(9000 + nfe)
        diff = sample_diffusion_from_initial(diffusion_model, ExperimentConfig(sample_steps=nfe), x_init, nfe)
        flow = sample_flow_from_initial(flow_model, x_init, nfe)
        diff_values.append(sliced_wasserstein_distance(diff, target, rng, projections=256))
        flow_values.append(sliced_wasserstein_distance(flow, target, rng, projections=256))

    width, height = 1080 * SCALE, 620 * SCALE
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img, "RGBA")
    rect = (110 * SCALE, 92 * SCALE, width - 90 * SCALE, height - 100 * SCALE)
    draw.text((30 * SCALE, 24 * SCALE), "NFE sensitivity", fill=INK, font=font())
    draw.text((30 * SCALE, 46 * SCALE), "same trained models; fewer function evaluations reveal solver-budget dependence", fill=MUTED, font=font())
    ymax = max(max(diff_values), max(flow_values)) * 1.12
    for frac in np.linspace(0.0, 1.0, 6):
        y = rect[3] - frac * (rect[3] - rect[1])
        draw.line((rect[0], y, rect[2], y), fill=GRID, width=1)
    draw.rectangle(rect, outline=(86, 94, 104, 255), width=2)

    def x_pos(nfe: int) -> float:
        logs = np.log(np.array(nfes, dtype=np.float64))
        return float(rect[0] + (np.log(nfe) - logs.min()) / (logs.max() - logs.min()) * (rect[2] - rect[0]))

    def y_pos(v: float) -> float:
        return float(rect[3] - v / ymax * (rect[3] - rect[1]))

    for nfe in nfes:
        x = x_pos(nfe)
        draw.line((x, rect[1], x, rect[3]), fill=(237, 239, 242, 255), width=1)
        draw.text((x - 16 * SCALE, rect[3] + 18 * SCALE), str(nfe), fill=MUTED, font=font())

    for values, color, label_y in [(diff_values, DIFFUSION, 112 * SCALE), (flow_values, FLOW, 142 * SCALE)]:
        pts = [(x_pos(nfe), y_pos(v)) for nfe, v in zip(nfes, values)]
        for a, b in zip(pts[:-1], pts[1:]):
            draw.line((a[0], a[1], b[0], b[1]), fill=color, width=4)
        for (x, y), v in zip(pts, values):
            r = 7 * SCALE
            draw.ellipse((x - r, y - r, x + r, y + r), fill=color)
            draw.text((x + 8 * SCALE, y - 18 * SCALE), f"{v:.3f}", fill=INK, font=font())
    draw.text((rect[0], rect[3] + 48 * SCALE), "NFE", fill=MUTED, font=font())
    draw.text((rect[0] - 82 * SCALE, rect[1] - 8 * SCALE), "SWD ↓", fill=MUTED, font=font())
    draw.rectangle((width - 300 * SCALE, 112 * SCALE, width - 280 * SCALE, 132 * SCALE), fill=DIFFUSION)
    draw.text((width - 270 * SCALE, 108 * SCALE), "Diffusion / DDIM", fill=INK, font=font())
    draw.rectangle((width - 300 * SCALE, 144 * SCALE, width - 280 * SCALE, 164 * SCALE), fill=FLOW)
    draw.text((width - 270 * SCALE, 140 * SCALE), "Flow matching / ODE", fill=INK, font=font())
    save_scaled(img, out_path)
    return {
        "nfe_values": nfes,
        "diffusion_swd_by_nfe": [float(v) for v in diff_values],
        "flow_swd_by_nfe": [float(v) for v in flow_values],
    }


def save_readme(out_dir: Path, metrics: dict) -> None:
    lines = [
        "# Analysis Figures",
        "",
        "这些图是为结果分析准备的候选素材，还没有放入 slide。",
        "",
        "- `01_vector_fields.png`: Diffusion 单步反向更新方向 vs Flow Matching 速度场。",
        "- `02_particle_trajectories.png`: 成对的相同初始粒子轨迹；左右同色表示同一个高斯起点。",
        "- `03_process_snapshots.png`: step 0/8/20/40/80 的静态过程切片。",
        "- `04_target_distance_hist.png`: 生成样本到目标样本的最近邻距离分布。",
        "- `05_stepwise_curves.png`: SWD、平均步长、平均转向角随步数变化。",
        "- `06_path_smoothness_hist.png`: 单粒子路径平均转向角分布，用于支撑“折线/平滑”的讨论。",
        "- `07_nfe_sensitivity.png`: 同一模型在不同 NFE 下的 SWD，用于观察有限步数近似的影响。",
        "",
        "补充指标：",
        "```json",
        json.dumps(metrics, indent=2, ensure_ascii=False),
        "```",
    ]
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    out_dir = ROOT / "outputs" / "analysis_figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = ExperimentConfig(seed=7, train_steps=2200, batch_size=512, sample_count=2500, sample_steps=80, frames=28)
    master_rng = np.random.default_rng(cfg.seed)
    diffusion_rng = np.random.default_rng(master_rng.integers(0, 2**32 - 1))
    flow_rng = np.random.default_rng(master_rng.integers(0, 2**32 - 1))
    eval_rng = np.random.default_rng(master_rng.integers(0, 2**32 - 1))

    print("training diffusion model for analysis figures")
    diffusion_model, _ = train_diffusion(cfg, diffusion_rng)
    print("training flow matching model for analysis figures")
    flow_model, _ = train_flow_matching(cfg, flow_rng)

    target = sample_s_curve(cfg.sample_count, eval_rng)
    _, diffusion_hist, flow_hist, _ = sample_with_history(diffusion_model, flow_model, cfg, eval_rng, cfg.sample_count)

    save_vector_field_panel(diffusion_model, flow_model, target, out_dir / "01_vector_fields.png")
    save_trajectory_panel(target, diffusion_hist, flow_hist, out_dir / "02_particle_trajectories.png")
    save_snapshot_strip(target, diffusion_hist, flow_hist, out_dir / "03_process_snapshots.png")
    distance_metrics = save_distance_histogram(target, diffusion_hist[-1], flow_hist[-1], out_dir / "04_target_distance_hist.png")
    curve_summary = save_step_curves(target, diffusion_hist, flow_hist, out_dir / "05_stepwise_curves.png")
    smoothness_summary = save_smoothness_histogram(diffusion_hist, flow_hist, out_dir / "06_path_smoothness_hist.png")
    nfe_summary = save_nfe_sensitivity(diffusion_model, flow_model, target, diffusion_hist[0], out_dir / "07_nfe_sensitivity.png")
    metrics = {**distance_metrics, **curve_summary, **smoothness_summary, **nfe_summary}
    save_readme(out_dir, metrics)
    print(json.dumps(metrics, indent=2))
    print(f"saved analysis figures to {out_dir}")


if __name__ == "__main__":
    main()
