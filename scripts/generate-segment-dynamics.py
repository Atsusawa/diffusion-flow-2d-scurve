from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scurve_diffusion_flow import ExperimentConfig, sample_s_curve, train_diffusion, train_flow_matching  # noqa: E402

ANALYSIS_PATH = ROOT / "scripts" / "generate-analysis-figures.py"
spec = importlib.util.spec_from_file_location("analysis_figures", ANALYSIS_PATH)
analysis = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(analysis)

SCALE = 2
INK = analysis.INK
MUTED = analysis.MUTED
GRID = analysis.GRID
DIFFUSION = analysis.DIFFUSION
FLOW = analysis.FLOW
ORANGE = analysis.TARGET_STRONG


def font(size: int = 12):
    return analysis.font(size)


def fixed_swd_series(hist: list[np.ndarray], target: np.ndarray, projections: int = 256) -> np.ndarray:
    rng = np.random.default_rng(24680)
    dirs = rng.normal(size=(projections, 2))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12
    target_proj = np.sort(target @ dirs.T, axis=0)
    values = []
    for state in hist:
        proj = np.sort(state @ dirs.T, axis=0)
        values.append(float(np.mean(np.abs(proj - target_proj))))
    return np.array(values, dtype=np.float64)


def turn_angles(hist: list[np.ndarray]) -> np.ndarray:
    points = np.stack(hist, axis=0)
    deltas = np.diff(points, axis=0)
    d0 = deltas[:-1]
    d1 = deltas[1:]
    denom = np.linalg.norm(d0, axis=2) * np.linalg.norm(d1, axis=2) + 1e-9
    cos = np.sum(d0 * d1, axis=2) / denom
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def segment_stats(hist: list[np.ndarray], swd: np.ndarray, segments: list[tuple[int, int]]) -> list[dict]:
    points = np.stack(hist, axis=0)
    deltas = np.diff(points, axis=0)
    step_len = np.linalg.norm(deltas, axis=2)
    angles = turn_angles(hist)
    radius = np.linalg.norm(points, axis=2).mean(axis=1)
    out = []
    for start, end in segments:
        seg_delta = points[end] - points[start]
        start_norm = np.linalg.norm(points[start], axis=1)
        radial = np.sum(seg_delta * points[start], axis=1) / (start_norm + 1e-9)
        angle_slice = angles[max(0, start) : max(0, end - 1)]
        out.append(
            {
                "segment": f"{start}-{end}",
                "swd_start": float(swd[start]),
                "swd_end": float(swd[end]),
                "swd_drop": float(swd[start] - swd[end]),
                "mean_step": float(step_len[start:end].mean()),
                "mean_turn_angle": float(angle_slice.mean()) if len(angle_slice) else None,
                "delta_radius": float(radius[end] - radius[start]),
                "mean_radial_projection": float(radial.mean()),
                "inward_fraction": float(np.mean(radial < 0.0)),
            }
        )
    return out


def plot_line(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    x: np.ndarray,
    series: list[tuple[str, np.ndarray, tuple[int, int, int, int]]],
    title: str,
    subtitle: str,
    y_min: float | None = None,
    y_max: float | None = None,
    segment_marks: list[int] | None = None,
) -> None:
    draw.text((rect[0], rect[1] - 58 * SCALE), title, fill=INK, font=font(18))
    draw.text((rect[0], rect[1] - 30 * SCALE), subtitle, fill=MUTED, font=font(11))
    all_values = np.concatenate([v for _, v, _ in series])
    ymin = float(all_values.min()) if y_min is None else y_min
    ymax = float(all_values.max()) if y_max is None else y_max
    pad = max((ymax - ymin) * 0.08, 1e-4)
    ymin -= pad
    ymax += pad

    for frac in np.linspace(0.0, 1.0, 5):
        yv = rect[3] - frac * (rect[3] - rect[1])
        draw.line((rect[0], yv, rect[2], yv), fill=GRID, width=1)
    for frac in np.linspace(0.0, 1.0, 6):
        xv = rect[0] + frac * (rect[2] - rect[0])
        draw.line((xv, rect[1], xv, rect[3]), fill=(238, 240, 243, 255), width=1)
    if segment_marks:
        for mark in segment_marks:
            xv = rect[0] + (mark - float(x.min())) / (float(x.max()) - float(x.min())) * (rect[2] - rect[0])
            draw.line((xv, rect[1], xv, rect[3]), fill=(166, 124, 74, 150), width=2)
            draw.text((xv + 5 * SCALE, rect[1] + 8 * SCALE), str(mark), fill=(120, 91, 58, 210), font=font(10))
    draw.rectangle(rect, outline=(88, 94, 104, 255), width=1)

    def pts(values: np.ndarray) -> list[tuple[float, float]]:
        xs = rect[0] + (x - float(x.min())) / (float(x.max()) - float(x.min())) * (rect[2] - rect[0])
        ys = rect[3] - (values - ymin) / (ymax - ymin) * (rect[3] - rect[1])
        return list(zip(xs, ys))

    for label, values, color in series:
        line = pts(values)
        for a, b in zip(line[:-1], line[1:]):
            draw.line((a[0], a[1], b[0], b[1]), fill=color, width=4 * SCALE)
        for step in [0, 8, 20, 40, 80]:
            px, py = line[step]
            r = 4 * SCALE
            draw.ellipse((px - r, py - r, px + r, py + r), fill=color)

    legend_x = rect[2] - 250 * SCALE
    for i, (label, _, color) in enumerate(series):
        y = rect[1] + (18 + i * 28) * SCALE
        draw.rectangle((legend_x, y, legend_x + 16 * SCALE, y + 16 * SCALE), fill=color)
        draw.text((legend_x + 24 * SCALE, y - 3 * SCALE), label, fill=INK, font=font(11))


def draw_segment_cards(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    diffusion: list[dict],
    flow: list[dict],
) -> None:
    draw.text((rect[0], rect[1] - 44 * SCALE), "Segment summary", fill=INK, font=font(18))
    draw.text((rect[0], rect[1] - 18 * SCALE), "larger SWD drop = stronger distribution alignment; negative radius = inward motion", fill=MUTED, font=font(11))
    segments = [d["segment"] for d in diffusion]
    gap = 14 * SCALE
    card_w = (rect[2] - rect[0] - gap * 3) / 4
    card_h = rect[3] - rect[1]
    for i, seg in enumerate(segments):
        x0 = rect[0] + i * (card_w + gap)
        x1 = x0 + card_w
        y0 = rect[1]
        y1 = y0 + card_h
        draw.rectangle((x0, y0, x1, y1), outline=(190, 182, 170, 255), fill=(255, 253, 249, 255), width=1)
        draw.text((x0 + 16 * SCALE, y0 + 12 * SCALE), seg, fill=INK, font=font(18))
        d = diffusion[i]
        f = flow[i]
        rows = [
            ("SWD drop", d["swd_drop"], f["swd_drop"], "{:+.3f}"),
            ("Δ radius", d["delta_radius"], f["delta_radius"], "{:+.3f}"),
            ("turn", d["mean_turn_angle"], f["mean_turn_angle"], "{:.1f}°"),
        ]
        yy = y0 + 52 * SCALE
        for label, dv, fv, fmt in rows:
            draw.text((x0 + 16 * SCALE, yy), label, fill=MUTED, font=font(11))
            d_text = "—" if dv is None else fmt.format(dv)
            f_text = "—" if fv is None else fmt.format(fv)
            draw.text((x0 + 16 * SCALE, yy + 22 * SCALE), d_text, fill=DIFFUSION, font=font(14))
            draw.text((x0 + card_w * 0.52, yy + 22 * SCALE), f_text, fill=FLOW, font=font(14))
            yy += 58 * SCALE


def save_segment_figure(
    diffusion_hist: list[np.ndarray],
    flow_hist: list[np.ndarray],
    target: np.ndarray,
    out_path: Path,
) -> dict:
    segments = [(0, 8), (8, 20), (20, 40), (40, 80)]
    swd_diff = fixed_swd_series(diffusion_hist, target)
    swd_flow = fixed_swd_series(flow_hist, target)
    diff_points = np.stack(diffusion_hist, axis=0)
    flow_points = np.stack(flow_hist, axis=0)
    radius_diff = np.linalg.norm(diff_points, axis=2).mean(axis=1)
    radius_flow = np.linalg.norm(flow_points, axis=2).mean(axis=1)
    diff_stats = segment_stats(diffusion_hist, swd_diff, segments)
    flow_stats = segment_stats(flow_hist, swd_flow, segments)

    width, height = 1500 * SCALE, 820 * SCALE
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img, "RGBA")
    draw.text((30 * SCALE, 22 * SCALE), "Segmented sampling dynamics", fill=INK, font=font(21))
    draw.text(
        (30 * SCALE, 52 * SCALE),
        "same initialization; segment boundaries are chosen from the visual snapshots: 0, 8, 20, 40, 80",
        fill=MUTED,
        font=font(11),
    )

    x = np.arange(81, dtype=np.float64)
    plot_line(
        draw,
        (92 * SCALE, 142 * SCALE, 720 * SCALE, 455 * SCALE),
        x,
        [("Diffusion", swd_diff, DIFFUSION), ("Flow", swd_flow, FLOW)],
        "Distance to target over sampling",
        "fixed-projection sliced Wasserstein distance",
        y_min=0.0,
        segment_marks=[8, 20, 40],
    )
    plot_line(
        draw,
        (840 * SCALE, 142 * SCALE, 1440 * SCALE, 455 * SCALE),
        x,
        [("Diffusion", radius_diff, DIFFUSION), ("Flow", radius_flow, FLOW)],
        "Mean radius over sampling",
        "reveals contraction vs local correction",
        segment_marks=[8, 20, 40],
    )
    draw_segment_cards(draw, (72 * SCALE, 560 * SCALE, 1440 * SCALE, 780 * SCALE), diff_stats, flow_stats)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)

    return {
        "segments": [f"{a}-{b}" for a, b in segments],
        "diffusion": diff_stats,
        "flow_matching": flow_stats,
        "swd_at_steps": {
            "steps": [0, 8, 20, 40, 80],
            "diffusion": [float(swd_diff[i]) for i in [0, 8, 20, 40, 80]],
            "flow_matching": [float(swd_flow[i]) for i in [0, 8, 20, 40, 80]],
        },
        "radius_at_steps": {
            "steps": [0, 8, 20, 40, 80],
            "diffusion": [float(radius_diff[i]) for i in [0, 8, 20, 40, 80]],
            "flow_matching": [float(radius_flow[i]) for i in [0, 8, 20, 40, 80]],
        },
    }


def main() -> None:
    out_dir = ROOT / "outputs" / "analysis_figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = ExperimentConfig(seed=7, train_steps=2200, batch_size=512, sample_count=2500, sample_steps=80, frames=28)
    master_rng = np.random.default_rng(cfg.seed)
    diffusion_rng = np.random.default_rng(master_rng.integers(0, 2**32 - 1))
    flow_rng = np.random.default_rng(master_rng.integers(0, 2**32 - 1))
    eval_rng = np.random.default_rng(master_rng.integers(0, 2**32 - 1))

    print("training diffusion model for segmented dynamics")
    diffusion_model, _ = train_diffusion(cfg, diffusion_rng)
    print("training flow matching model for segmented dynamics")
    flow_model, _ = train_flow_matching(cfg, flow_rng)

    target = sample_s_curve(cfg.sample_count, eval_rng)
    _, diffusion_hist, flow_hist, _ = analysis.sample_with_history(
        diffusion_model,
        flow_model,
        cfg,
        eval_rng,
        cfg.sample_count,
    )

    metrics = save_segment_figure(diffusion_hist, flow_hist, target, out_dir / "09_segment_dynamics.png")
    metrics_path = out_dir / "09_segment_dynamics_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"saved {out_dir / '09_segment_dynamics.png'}")


if __name__ == "__main__":
    main()
