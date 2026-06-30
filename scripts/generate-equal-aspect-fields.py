from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scurve_diffusion_flow import ExperimentConfig, sample_s_curve, to_pixel, train_diffusion, train_flow_matching  # noqa: E402

FOCUS_PATH = ROOT / "scripts" / "generate-dynamics-focus.py"
spec = importlib.util.spec_from_file_location("dynamics_focus", FOCUS_PATH)
focus = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(focus)

SCALE = 2
BOUNDS = focus.BOUNDS
INK = focus.INK
GRID = focus.GRID
AXIS = focus.AXIS
TARGET = focus.TARGET
DIFFUSION = focus.DIFFUSION
FLOW = focus.FLOW


def draw_equal_base(draw: ImageDraw.ImageDraw, rect: tuple[int, int, int, int]) -> None:
    for frac in np.linspace(0.0, 1.0, 6):
        x = rect[0] + frac * (rect[2] - rect[0])
        y = rect[1] + frac * (rect[3] - rect[1])
        draw.line((x, rect[1], x, rect[3]), fill=GRID, width=1)
        draw.line((rect[0], y, rect[2], y), fill=GRID, width=1)
    zero = to_pixel(np.array([[0.0, 0.0]]), BOUNDS, rect)[0]
    draw.line((zero[0], rect[1], zero[0], rect[3]), fill=AXIS, width=1)
    draw.line((rect[0], zero[1], rect[2], zero[1]), fill=AXIS, width=1)
    draw.rectangle(rect, outline=(88, 94, 104, 255), width=1)


def draw_field(
    target: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    color: tuple[int, int, int, int],
    out_path: Path,
) -> None:
    width, height = 1180 * SCALE, 1080 * SCALE
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img, "RGBA")

    xmin, xmax, ymin, ymax = BOUNDS
    data_ratio = (xmax - xmin) / (ymax - ymin)
    rect_h = 960 * SCALE
    rect_w = int(round(rect_h * data_ratio))
    left = (width - rect_w) // 2
    top = 60 * SCALE
    rect = (left, top, left + rect_w, top + rect_h)

    draw_equal_base(draw, rect)
    focus.draw_points(draw, target, rect, TARGET, radius=1, limit=900)
    focus.draw_points(draw, start, rect, (96, 105, 120, 44), radius=1, limit=1000)
    focus.draw_points(draw, end, rect, color[:3] + (112,), radius=1, limit=1000)

    for p0, delta, _ in focus.binned_displacements(start, end):
        p1 = p0 + 2.2 * delta
        pix = to_pixel(np.stack([p0, p1], axis=0), BOUNDS, rect)
        a = (float(pix[0, 0]), float(pix[0, 1]))
        b = (float(pix[1, 0]), float(pix[1, 1]))
        draw.line((a[0], a[1], b[0], b[1]), fill=color, width=3 * SCALE)
        focus.draw_arrow_head(draw, a, b, color, size=8.5 * SCALE)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def main() -> None:
    cfg = ExperimentConfig(seed=7, train_steps=2200, batch_size=512, sample_count=2500, sample_steps=80, frames=28)
    master_rng = np.random.default_rng(cfg.seed)
    diffusion_rng = np.random.default_rng(master_rng.integers(0, 2**32 - 1))
    flow_rng = np.random.default_rng(master_rng.integers(0, 2**32 - 1))
    eval_rng = np.random.default_rng(master_rng.integers(0, 2**32 - 1))

    print("training diffusion model for equal-aspect fields")
    diffusion_model, _ = train_diffusion(cfg, diffusion_rng)
    print("training flow matching model for equal-aspect fields")
    flow_model, _ = train_flow_matching(cfg, flow_rng)

    target = sample_s_curve(cfg.sample_count, eval_rng)
    start, diffusion_hist, flow_hist, _ = focus.analysis.sample_with_history(
        diffusion_model,
        flow_model,
        cfg,
        eval_rng,
        cfg.sample_count,
    )

    outputs = ROOT / "outputs" / "analysis_figures"
    assets = ROOT / "presentation_v2" / "assets"
    for directory in (outputs, assets):
        draw_field(target, start, diffusion_hist[8], DIFFUSION, directory / "08_early_diffusion_field_equal.png")
        draw_field(target, start, flow_hist[8], FLOW, directory / "08_early_flow_field_equal.png")
    print("saved equal-aspect field panels")


if __name__ == "__main__":
    main()
