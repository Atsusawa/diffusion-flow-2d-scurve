from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scurve_diffusion_flow import ExperimentConfig, sample_s_curve, train_diffusion, train_flow_matching, to_pixel  # noqa: E402

ANALYSIS_PATH = ROOT / "scripts" / "generate-analysis-figures.py"

spec = importlib.util.spec_from_file_location("analysis_figures", ANALYSIS_PATH)
analysis = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(analysis)

SCALE = 2
BOUNDS = analysis.BOUNDS
INK = analysis.INK
MUTED = analysis.MUTED
GRID = analysis.GRID
AXIS = analysis.AXIS
TARGET = analysis.TARGET
DIFFUSION = analysis.DIFFUSION
FLOW = analysis.FLOW


def font(size: int = 12):
    return analysis.font(size)


def draw_base(draw: ImageDraw.ImageDraw, rect: tuple[int, int, int, int], title: str, subtitle: str) -> None:
    draw.text((rect[0], rect[1] - 54 * SCALE), title, fill=INK, font=font(17))
    draw.text((rect[0], rect[1] - 30 * SCALE), subtitle, fill=MUTED, font=font(11))
    for frac in np.linspace(0.0, 1.0, 6):
        x = rect[0] + frac * (rect[2] - rect[0])
        y = rect[1] + frac * (rect[3] - rect[1])
        draw.line((x, rect[1], x, rect[3]), fill=GRID, width=1)
        draw.line((rect[0], y, rect[2], y), fill=GRID, width=1)
    zero = to_pixel(np.array([[0.0, 0.0]]), BOUNDS, rect)[0]
    draw.line((zero[0], rect[1], zero[0], rect[3]), fill=AXIS, width=1)
    draw.line((rect[0], zero[1], rect[2], zero[1]), fill=AXIS, width=1)
    draw.rectangle(rect, outline=(88, 94, 104, 255), width=1)


def draw_points(
    draw: ImageDraw.ImageDraw,
    points: np.ndarray,
    rect: tuple[int, int, int, int],
    color: tuple[int, int, int, int],
    radius: int = 1,
    limit: int | None = None,
) -> None:
    if limit is not None and len(points) > limit:
        points = points[:limit]
    pixels = to_pixel(points, BOUNDS, rect)
    r = max(1, radius * SCALE)
    for x, y in pixels:
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color)


def draw_arrow_head(
    draw: ImageDraw.ImageDraw,
    p0: tuple[float, float],
    p1: tuple[float, float],
    color: tuple[int, int, int, int],
    size: float = 7.5,
) -> None:
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    n = float(np.hypot(dx, dy))
    if n < 1e-6:
        return
    ux, uy = dx / n, dy / n
    px, py = -uy, ux
    a = (p1[0] - size * ux + 0.5 * size * px, p1[1] - size * uy + 0.5 * size * py)
    b = (p1[0] - size * ux - 0.5 * size * px, p1[1] - size * uy - 0.5 * size * py)
    draw.polygon([p1, a, b], fill=color)


def binned_displacements(start: np.ndarray, end: np.ndarray, bins_x: int = 13, bins_y: int = 11) -> list[tuple[np.ndarray, np.ndarray, int]]:
    xmin, xmax, ymin, ymax = BOUNDS
    x_edges = np.linspace(xmin + 0.25, xmax - 0.25, bins_x + 1)
    y_edges = np.linspace(ymin + 0.25, ymax - 0.25, bins_y + 1)
    out: list[tuple[np.ndarray, np.ndarray, int]] = []
    for ix in range(bins_x):
        for iy in range(bins_y):
            mask = (
                (start[:, 0] >= x_edges[ix])
                & (start[:, 0] < x_edges[ix + 1])
                & (start[:, 1] >= y_edges[iy])
                & (start[:, 1] < y_edges[iy + 1])
            )
            count = int(mask.sum())
            if count < 7:
                continue
            p0 = start[mask].mean(axis=0)
            delta = (end[mask] - start[mask]).mean(axis=0)
            out.append((p0, delta, count))
    return out


def draw_displacement_panel(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    title: str,
    subtitle: str,
    target: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    color: tuple[int, int, int, int],
) -> None:
    draw_base(draw, rect, title, subtitle)
    draw_points(draw, target, rect, TARGET, radius=1, limit=900)
    draw_points(draw, start, rect, (96, 105, 120, 44), radius=1, limit=1000)
    draw_points(draw, end, rect, color[:3] + (112,), radius=1, limit=1000)
    for p0, delta, _ in binned_displacements(start, end):
        p1 = p0 + 2.2 * delta
        pix = to_pixel(np.stack([p0, p1], axis=0), BOUNDS, rect)
        a = (float(pix[0, 0]), float(pix[0, 1]))
        b = (float(pix[1, 0]), float(pix[1, 1]))
        draw.line((a[0], a[1], b[0], b[1]), fill=color, width=3 * SCALE)
        draw_arrow_head(draw, a, b, color, size=8.5 * SCALE)


def radial_metrics(start: np.ndarray, hist: list[np.ndarray], step: int) -> dict[str, float]:
    r0 = np.linalg.norm(start, axis=1)
    cur = hist[step]
    r_cur = np.linalg.norm(cur, axis=1)
    radial_projection = np.sum((cur - start) * start, axis=1) / (r0 + 1e-9)
    return {
        "delta_mean_radius": float(np.mean(r_cur - r0)),
        "mean_radial_projection": float(np.mean(radial_projection)),
        "inward_fraction": float(np.mean(radial_projection < 0.0)),
        "mean_radius_step": float(np.mean(r_cur)),
    }


def draw_radius_curve(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    start: np.ndarray,
    diffusion_hist: list[np.ndarray],
    flow_hist: list[np.ndarray],
    upto: int = 20,
) -> None:
    draw.text((rect[0], rect[1] - 42 * SCALE), "Mean radius in early steps", fill=INK, font=font(17))
    draw.text((rect[0], rect[1] - 18 * SCALE), "radius = distance to origin; first 20 of 80 sampling steps", fill=MUTED, font=font(11))
    r_diff = np.array([np.linalg.norm(diffusion_hist[i], axis=1).mean() for i in range(upto + 1)])
    r_flow = np.array([np.linalg.norm(flow_hist[i], axis=1).mean() for i in range(upto + 1)])
    ymin = min(float(r_diff.min()), float(r_flow.min())) - 0.03
    ymax = max(float(r_diff.max()), float(r_flow.max())) + 0.03
    for frac in np.linspace(0.0, 1.0, 5):
        y = rect[3] - frac * (rect[3] - rect[1])
        draw.line((rect[0], y, rect[2], y), fill=GRID, width=1)
    for frac in np.linspace(0.0, 1.0, 6):
        x = rect[0] + frac * (rect[2] - rect[0])
        draw.line((x, rect[1], x, rect[3]), fill=(235, 237, 240, 255), width=1)
    draw.rectangle(rect, outline=(88, 94, 104, 255), width=1)

    def pts(values: np.ndarray) -> list[tuple[float, float]]:
        xs = np.linspace(rect[0], rect[2], len(values))
        ys = rect[3] - (values - ymin) / (ymax - ymin) * (rect[3] - rect[1])
        return list(zip(xs, ys))

    for values, color in [(r_diff, DIFFUSION), (r_flow, FLOW)]:
        points = pts(values)
        for a, b in zip(points[:-1], points[1:]):
            draw.line((a[0], a[1], b[0], b[1]), fill=color, width=4 * SCALE)
        for k in [0, 8, 20]:
            x, y = points[k]
            r = 4 * SCALE
            draw.ellipse((x - r, y - r, x + r, y + r), fill=color)
    draw.text((rect[0], rect[3] + 12 * SCALE), "0", fill=MUTED, font=font(10))
    draw.text((rect[0] + (rect[2] - rect[0]) * 8 / 20 - 10 * SCALE, rect[3] + 12 * SCALE), "8", fill=MUTED, font=font(10))
    draw.text((rect[2] - 20 * SCALE, rect[3] + 12 * SCALE), "20", fill=MUTED, font=font(10))
    draw.text((rect[0] - 56 * SCALE, rect[1] - 4 * SCALE), f"{ymax:.2f}", fill=MUTED, font=font(10))
    draw.text((rect[0] - 56 * SCALE, rect[3] - 8 * SCALE), f"{ymin:.2f}", fill=MUTED, font=font(10))
    draw.rectangle((rect[2] - 230 * SCALE, rect[1] + 14 * SCALE, rect[2] - 215 * SCALE, rect[1] + 29 * SCALE), fill=DIFFUSION)
    draw.text((rect[2] - 208 * SCALE, rect[1] + 10 * SCALE), "Diffusion", fill=INK, font=font(11))
    draw.rectangle((rect[2] - 116 * SCALE, rect[1] + 14 * SCALE, rect[2] - 101 * SCALE, rect[1] + 29 * SCALE), fill=FLOW)
    draw.text((rect[2] - 94 * SCALE, rect[1] + 10 * SCALE), "Flow", fill=INK, font=font(11))


def save_focus_figure(
    target: np.ndarray,
    start: np.ndarray,
    diffusion_hist: list[np.ndarray],
    flow_hist: list[np.ndarray],
    out_path: Path,
    step: int = 8,
) -> dict:
    width, height = 1500 * SCALE, 760 * SCALE
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img, "RGBA")
    draw.text((28 * SCALE, 20 * SCALE), "Early displacement: step 0 -> 8", fill=INK, font=font(20))
    draw.text((28 * SCALE, 48 * SCALE), "grey = initial Gaussian; colored = step 8; arrows show binned mean displacement, magnified for direction", fill=MUTED, font=font(11))

    left_rect = (72 * SCALE, 128 * SCALE, 710 * SCALE, 490 * SCALE)
    right_rect = (790 * SCALE, 128 * SCALE, 1428 * SCALE, 490 * SCALE)
    curve_rect = (118 * SCALE, 590 * SCALE, 1428 * SCALE, 710 * SCALE)

    draw_displacement_panel(
        draw,
        left_rect,
        "Diffusion / DDIM",
        "mixed early displacement; no global inward field",
        target,
        start,
        diffusion_hist[step],
        DIFFUSION,
    )
    draw_displacement_panel(
        draw,
        right_rect,
        "Flow Matching / ODE",
        "near t=0, velocity points toward the target mean",
        target,
        start,
        flow_hist[step],
        FLOW,
    )
    draw_radius_curve(draw, curve_rect, start, diffusion_hist, flow_hist, upto=20)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)

    return {
        "step": step,
        "diffusion": radial_metrics(start, diffusion_hist, step),
        "flow_matching": radial_metrics(start, flow_hist, step),
    }


def save_slide_focus_figure(
    target: np.ndarray,
    start: np.ndarray,
    diffusion_hist: list[np.ndarray],
    flow_hist: list[np.ndarray],
    out_path: Path,
    step: int = 8,
) -> None:
    """A tighter crop for the HTML slide: no global title, larger evidence panels."""
    width, height = 1320 * SCALE, 790 * SCALE
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img, "RGBA")

    left_rect = (52 * SCALE, 82 * SCALE, 620 * SCALE, 450 * SCALE)
    right_rect = (700 * SCALE, 82 * SCALE, 1268 * SCALE, 450 * SCALE)
    curve_rect = (96 * SCALE, 605 * SCALE, 1268 * SCALE, 748 * SCALE)

    draw_displacement_panel(
        draw,
        left_rect,
        "Diffusion / DDIM",
        "step 0 -> 8: mixed local updates",
        target,
        start,
        diffusion_hist[step],
        DIFFUSION,
    )
    draw_displacement_panel(
        draw,
        right_rect,
        "Flow Matching / ODE",
        "step 0 -> 8: coherent inward transport",
        target,
        start,
        flow_hist[step],
        FLOW,
    )
    draw_radius_curve(draw, curve_rect, start, diffusion_hist, flow_hist, upto=20)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def main() -> None:
    out_dir = ROOT / "outputs" / "analysis_figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = ExperimentConfig(seed=7, train_steps=2200, batch_size=512, sample_count=2500, sample_steps=80, frames=28)
    master_rng = np.random.default_rng(cfg.seed)
    diffusion_rng = np.random.default_rng(master_rng.integers(0, 2**32 - 1))
    flow_rng = np.random.default_rng(master_rng.integers(0, 2**32 - 1))
    eval_rng = np.random.default_rng(master_rng.integers(0, 2**32 - 1))

    print("training diffusion model for focused dynamics figure")
    diffusion_model, _ = train_diffusion(cfg, diffusion_rng)
    print("training flow matching model for focused dynamics figure")
    flow_model, _ = train_flow_matching(cfg, flow_rng)

    target = sample_s_curve(cfg.sample_count, eval_rng)
    start, diffusion_hist, flow_hist, _ = analysis.sample_with_history(
        diffusion_model,
        flow_model,
        cfg,
        eval_rng,
        cfg.sample_count,
    )
    metrics = save_focus_figure(target, start, diffusion_hist, flow_hist, out_dir / "08_early_dynamics_focus.png")
    save_slide_focus_figure(target, start, diffusion_hist, flow_hist, out_dir / "08_early_dynamics_focus_slide.png")
    metrics_path = out_dir / "08_early_dynamics_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"saved {out_dir / '08_early_dynamics_focus.png'}")


if __name__ == "__main__":
    main()
