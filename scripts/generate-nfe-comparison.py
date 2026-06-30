from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scurve_diffusion_flow import (  # noqa: E402
    ExperimentConfig,
    MLP,
    diffusion_alpha_sigma,
    make_model,
    sample_s_curve,
    time_features,
    to_pixel,
    train_diffusion,
    train_flow_matching,
)

ANALYSIS_PATH = ROOT / "scripts" / "generate-analysis-figures.py"
spec = importlib.util.spec_from_file_location("analysis_figures", ANALYSIS_PATH)
analysis = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(analysis)

Array = np.ndarray
SCALE = 2
BOUNDS = (-2.65, 2.65, -2.45, 2.45)

INK = (27, 33, 44, 255)
MUTED = (86, 95, 110, 255)
GRID = (224, 228, 232, 255)
AXIS = (178, 185, 196, 255)
DIFFUSION = (47, 102, 179, 220)
FLOW = (46, 139, 95, 225)
TARGET = (220, 127, 44, 105)


def cfg_signature(cfg: ExperimentConfig) -> dict:
    return {
        "seed": cfg.seed,
        "train_steps": cfg.train_steps,
        "batch_size": cfg.batch_size,
        "hidden_size": cfg.hidden_size,
        "hidden_layers": cfg.hidden_layers,
        "learning_rate": cfg.learning_rate,
        "diffusion_t_min": cfg.diffusion_t_min,
        "diffusion_t_max": cfg.diffusion_t_max,
    }


def model_state(model: MLP) -> dict[str, Array]:
    state: dict[str, Array] = {}
    for i, (w, b) in enumerate(zip(model.weights, model.biases)):
        state[f"w{i}"] = w
        state[f"b{i}"] = b
    return state


def load_model_state(model: MLP, data: np.lib.npyio.NpzFile) -> None:
    for i in range(len(model.weights)):
        model.weights[i][...] = data[f"w{i}"]
        model.biases[i][...] = data[f"b{i}"]


def save_model_bundle(
    path: Path,
    diffusion_model: MLP,
    flow_model: MLP,
    cfg: ExperimentConfig,
    training_metrics: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Array] = {}
    for key, value in model_state(diffusion_model).items():
        arrays[f"diffusion_{key}"] = value
    for key, value in model_state(flow_model).items():
        arrays[f"flow_{key}"] = value
    metadata = {
        "config": cfg_signature(cfg),
        "training": training_metrics,
    }
    arrays["metadata_json"] = np.array(json.dumps(metadata, ensure_ascii=False))
    np.savez(path, **arrays)


def load_model_bundle(path: Path, cfg: ExperimentConfig) -> tuple[MLP, MLP, dict] | None:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"]))
        if metadata.get("config") != cfg_signature(cfg):
            return None
        seed_rng = np.random.default_rng(cfg.seed)
        diffusion_model = make_model(cfg, np.random.default_rng(seed_rng.integers(0, 2**32 - 1)))
        flow_model = make_model(cfg, np.random.default_rng(seed_rng.integers(0, 2**32 - 1)))
        diffusion_data = {k.replace("diffusion_", ""): data[k] for k in data.files if k.startswith("diffusion_")}
        flow_data = {k.replace("flow_", ""): data[k] for k in data.files if k.startswith("flow_")}

        class BundleView:
            def __init__(self, values: dict[str, Array]) -> None:
                self.values = values

            def __getitem__(self, key: str) -> Array:
                return self.values[key]

        load_model_state(diffusion_model, BundleView(diffusion_data))  # type: ignore[arg-type]
        load_model_state(flow_model, BundleView(flow_data))  # type: ignore[arg-type]
        return diffusion_model, flow_model, metadata


def get_or_train_models(cfg: ExperimentConfig, model_path: Path) -> tuple[MLP, MLP, dict, str]:
    loaded = load_model_bundle(model_path, cfg)
    if loaded is not None:
        diffusion_model, flow_model, metadata = loaded
        return diffusion_model, flow_model, metadata["training"], "loaded"

    master_rng = np.random.default_rng(cfg.seed)
    diffusion_rng = np.random.default_rng(master_rng.integers(0, 2**32 - 1))
    flow_rng = np.random.default_rng(master_rng.integers(0, 2**32 - 1))

    print("training diffusion model for reusable NFE comparison weights")
    diffusion_model, diffusion_train = train_diffusion(cfg, diffusion_rng)
    print("training flow matching model for reusable NFE comparison weights")
    flow_model, flow_train = train_flow_matching(cfg, flow_rng)
    training_metrics = {
        "diffusion": diffusion_train,
        "flow_matching": flow_train,
    }
    save_model_bundle(model_path, diffusion_model, flow_model, cfg, training_metrics)
    return diffusion_model, flow_model, training_metrics, "trained"


def get_or_create_target(cfg: ExperimentConfig, target_path: Path, target_seed: int) -> tuple[Array, str]:
    if target_path.exists():
        with np.load(target_path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"]))
            if metadata.get("sample_count") == cfg.sample_count and metadata.get("target_seed") == target_seed:
                return data["target"].copy(), "loaded"
    rng = np.random.default_rng(target_seed)
    target = sample_s_curve(cfg.sample_count, rng)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "sample_count": cfg.sample_count,
        "target_seed": target_seed,
    }
    np.savez(target_path, target=target, metadata_json=np.array(json.dumps(metadata, ensure_ascii=False)))
    return target, "created"


def font(size: int = 12, serif: bool = False) -> ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/georgia.ttf") if serif else Path("C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/times.ttf") if serif else Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/msyh.ttc"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size * SCALE)
    return ImageFont.load_default()


def cjk_font(size: int = 12) -> ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simsun.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size * SCALE)
    return font(size)


def predict_diffusion_step(model: MLP, x: Array, t_cur: float, t_next: float) -> Array:
    t_batch = np.full((len(x), 1), t_cur, dtype=np.float64)
    eps_pred = model.predict(time_features(x, t_batch))
    alpha, sigma = diffusion_alpha_sigma(t_cur)
    alpha_next, sigma_next = diffusion_alpha_sigma(t_next)
    x0_hat = (x - sigma * eps_pred) / max(float(alpha), 1e-3)
    x0_hat = np.clip(x0_hat, -4.0, 4.0)
    return alpha_next * x0_hat + sigma_next * eps_pred


def sample_diffusion_from_initial(model: MLP, cfg: ExperimentConfig, x_init: Array, nfe: int) -> Array:
    x = x_init.copy()
    times = np.linspace(cfg.diffusion_t_max, 0.0, nfe + 1)
    for i in range(nfe):
        x = predict_diffusion_step(model, x, float(times[i]), float(times[i + 1]))
    return x


def sample_flow_from_initial(model: MLP, x_init: Array, nfe: int) -> Array:
    x = x_init.copy()
    times = np.linspace(0.0, 1.0, nfe + 1)
    for i in range(nfe):
        t_cur = float(times[i])
        dt = float(times[i + 1] - times[i])
        t_batch = np.full((len(x), 1), t_cur, dtype=np.float64)
        x = x + dt * model.predict(time_features(x, t_batch))
    return x


def make_fixed_swd(target: Array, projections: int = 384, seed: int = 24680):
    rng = np.random.default_rng(seed)
    dirs = rng.normal(size=(projections, 2))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12
    target_proj = np.sort(target @ dirs.T, axis=0)

    def metric(points: Array) -> float:
        proj = np.sort(points @ dirs.T, axis=0)
        return float(np.mean(np.abs(proj - target_proj)))

    return metric


def nearest_distances(points: Array, target: Array, chunk: int = 512) -> Array:
    parts = []
    for i in range(0, len(points), chunk):
        p = points[i : i + chunk]
        d2 = np.sum((p[:, None, :] - target[None, :, :]) ** 2, axis=2)
        parts.append(np.sqrt(np.min(d2, axis=1)))
    return np.concatenate(parts)


def summarize(values: list[float]) -> dict:
    arr = np.array(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def evaluate_nfe(
    diffusion_model: MLP,
    flow_model: MLP,
    cfg: ExperimentConfig,
    target: Array,
    nfes: list[int],
    eval_seeds: list[int],
) -> tuple[dict, dict[tuple[str, int], Array]]:
    swd = make_fixed_swd(target)
    raw: dict[str, dict[str, list[float]]] = {}
    snapshot_samples: dict[tuple[str, int], Array] = {}

    for nfe in nfes:
        for method in ["diffusion", "flow_matching"]:
            raw.setdefault(method, {}).setdefault(str(nfe), [])

    timing: dict[str, dict[str, list[float]]] = {
        "diffusion": {str(nfe): [] for nfe in nfes},
        "flow_matching": {str(nfe): [] for nfe in nfes},
    }
    nn_median: dict[str, dict[str, list[float]]] = {
        "diffusion": {str(nfe): [] for nfe in nfes},
        "flow_matching": {str(nfe): [] for nfe in nfes},
    }
    nn_p90: dict[str, dict[str, list[float]]] = {
        "diffusion": {str(nfe): [] for nfe in nfes},
        "flow_matching": {str(nfe): [] for nfe in nfes},
    }

    for seed_idx, seed in enumerate(eval_seeds):
        rng = np.random.default_rng(seed)
        x_init = rng.normal(size=(cfg.sample_count, 2))
        for nfe in nfes:
            start = time.perf_counter()
            diffusion_samples = sample_diffusion_from_initial(diffusion_model, cfg, x_init, nfe)
            timing["diffusion"][str(nfe)].append(time.perf_counter() - start)

            start = time.perf_counter()
            flow_samples = sample_flow_from_initial(flow_model, x_init, nfe)
            timing["flow_matching"][str(nfe)].append(time.perf_counter() - start)

            for method, samples in [("diffusion", diffusion_samples), ("flow_matching", flow_samples)]:
                raw[method][str(nfe)].append(swd(samples))
                d = nearest_distances(samples, target)
                nn_median[method][str(nfe)].append(float(np.median(d)))
                nn_p90[method][str(nfe)].append(float(np.percentile(d, 90)))
                if seed_idx == 0:
                    snapshot_samples[(method, nfe)] = samples.copy()

    metrics = {
        "nfe_values": nfes,
        "eval_seeds": eval_seeds,
        "sample_count": cfg.sample_count,
        "diffusion": {},
        "flow_matching": {},
        "notes": {
            "nfe": "network function evaluations during sampling; both samplers call one network once per step",
            "swd": "fixed-projection sliced Wasserstein distance; lower is closer to target distribution",
            "nearest_distance": "Euclidean distance from each generated point to the nearest target sample",
        },
    }
    for method in ["diffusion", "flow_matching"]:
        for nfe in nfes:
            key = str(nfe)
            metrics[method][key] = {
                "swd": summarize(raw[method][key]),
                "nearest_median": summarize(nn_median[method][key]),
                "nearest_p90": summarize(nn_p90[method][key]),
                "sample_seconds": summarize(timing[method][key]),
            }
    return metrics, snapshot_samples


def line_points(
    nfes: list[int],
    values: list[float],
    rect: tuple[int, int, int, int],
    y_min: float,
    y_max: float,
) -> list[tuple[float, float]]:
    logs = np.log(np.array(nfes, dtype=np.float64))
    x_min, x_max = float(logs.min()), float(logs.max())
    pts = []
    for nfe, value in zip(nfes, values):
        x = rect[0] + (np.log(nfe) - x_min) / (x_max - x_min) * (rect[2] - rect[0])
        y = rect[3] - (value - y_min) / (y_max - y_min) * (rect[3] - rect[1])
        pts.append((float(x), float(y)))
    return pts


def draw_metric_curve(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    nfes: list[int],
    diff_values: list[float],
    flow_values: list[float],
    diff_std: list[float],
    flow_std: list[float],
    title: str,
    y_label: str,
) -> None:
    all_y = np.array(diff_values + flow_values + [0.0], dtype=np.float64)
    y_min = 0.0
    y_max = float(max(all_y.max() * 1.18, 0.01))
    draw.text((rect[0], rect[1] - 54 * SCALE), title, fill=INK, font=font(18, serif=True))
    draw.text((rect[0], rect[1] - 26 * SCALE), y_label, fill=MUTED, font=font(10))

    for frac in np.linspace(0.0, 1.0, 6):
        y = rect[3] - frac * (rect[3] - rect[1])
        draw.line((rect[0], y, rect[2], y), fill=GRID, width=1)
    logs = np.log(np.array(nfes, dtype=np.float64))
    for nfe, lx in zip(nfes, logs):
        x = rect[0] + (lx - logs.min()) / (logs.max() - logs.min()) * (rect[2] - rect[0])
        draw.line((x, rect[1], x, rect[3]), fill=(238, 240, 243, 255), width=1)
        label = str(nfe)
        draw.text((x - 10 * SCALE, rect[3] + 12 * SCALE), label, fill=MUTED, font=font(9))
    draw.rectangle(rect, outline=(88, 94, 104, 255), width=1)

    for values, stds, color in [(diff_values, diff_std, DIFFUSION), (flow_values, flow_std, FLOW)]:
        pts = line_points(nfes, values, rect, y_min, y_max)
        upper = line_points(nfes, [v + s for v, s in zip(values, stds)], rect, y_min, y_max)
        lower = line_points(nfes, [max(0.0, v - s) for v, s in zip(values, stds)], rect, y_min, y_max)
        band = upper + list(reversed(lower))
        draw.polygon(band, fill=(color[0], color[1], color[2], 36))
        for a, b in zip(pts[:-1], pts[1:]):
            draw.line((a[0], a[1], b[0], b[1]), fill=color, width=4 * SCALE)
        for x, y in pts:
            r = 5 * SCALE
            draw.ellipse((x - r, y - r, x + r, y + r), fill=color, outline=(255, 255, 255, 220), width=1 * SCALE)


def save_nfe_curve(metrics: dict, out_path: Path) -> None:
    nfes = metrics["nfe_values"]
    width, height = 1420 * SCALE, 680 * SCALE
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img, "RGBA")
    draw.text((32 * SCALE, 24 * SCALE), "NFE sensitivity", fill=INK, font=font(24, serif=True))
    draw.text((32 * SCALE, 58 * SCALE), "same trained models, same target distribution, fixed SWD projections", fill=MUTED, font=font(11))

    rect1 = (90 * SCALE, 144 * SCALE, 680 * SCALE, 560 * SCALE)
    rect2 = (805 * SCALE, 144 * SCALE, 1348 * SCALE, 560 * SCALE)

    diff_swd = [metrics["diffusion"][str(n)]["swd"]["mean"] for n in nfes]
    flow_swd = [metrics["flow_matching"][str(n)]["swd"]["mean"] for n in nfes]
    diff_swd_std = [metrics["diffusion"][str(n)]["swd"]["std"] for n in nfes]
    flow_swd_std = [metrics["flow_matching"][str(n)]["swd"]["std"] for n in nfes]
    draw_metric_curve(draw, rect1, nfes, diff_swd, flow_swd, diff_swd_std, flow_swd_std, "Distribution distance", "fixed-projection SWD, lower is better")

    diff_nn = [metrics["diffusion"][str(n)]["nearest_p90"]["mean"] for n in nfes]
    flow_nn = [metrics["flow_matching"][str(n)]["nearest_p90"]["mean"] for n in nfes]
    diff_nn_std = [metrics["diffusion"][str(n)]["nearest_p90"]["std"] for n in nfes]
    flow_nn_std = [metrics["flow_matching"][str(n)]["nearest_p90"]["std"] for n in nfes]
    draw_metric_curve(draw, rect2, nfes, diff_nn, flow_nn, diff_nn_std, flow_nn_std, "Tail fit to S-curve", "p90 nearest-target distance, lower is better")

    legend_x = width - 316 * SCALE
    legend_y = 28 * SCALE
    draw.rectangle((legend_x, legend_y, legend_x + 18 * SCALE, legend_y + 18 * SCALE), fill=DIFFUSION)
    draw.text((legend_x + 28 * SCALE, legend_y - 2 * SCALE), "Diffusion / deterministic DDIM-like", fill=INK, font=font(10))
    draw.rectangle((legend_x, legend_y + 30 * SCALE, legend_x + 18 * SCALE, legend_y + 48 * SCALE), fill=FLOW)
    draw.text((legend_x + 28 * SCALE, legend_y + 28 * SCALE), "Flow Matching / Euler ODE", fill=INK, font=font(10))
    draw.text((rect1[2] - 20 * SCALE, rect1[3] + 46 * SCALE), "NFE", fill=MUTED, font=font(10))
    draw.text((rect2[2] - 20 * SCALE, rect2[3] + 46 * SCALE), "NFE", fill=MUTED, font=font(10))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def draw_metric_curve_slide(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    nfes: list[int],
    diff_values: list[float],
    flow_values: list[float],
    diff_std: list[float],
    flow_std: list[float],
    title: str,
    y_label: str,
) -> None:
    all_y = np.array(diff_values + flow_values + [0.0], dtype=np.float64)
    y_min = 0.0
    y_max = float(max(all_y.max() * 1.18, 0.01))
    draw.text((rect[0], rect[1] - 58 * SCALE), title, fill=INK, font=cjk_font(20))
    draw.text((rect[0], rect[1] - 25 * SCALE), y_label, fill=MUTED, font=font(11))

    for frac in np.linspace(0.0, 1.0, 6):
        y = rect[3] - frac * (rect[3] - rect[1])
        draw.line((rect[0], y, rect[2], y), fill=GRID, width=1)
    logs = np.log(np.array(nfes, dtype=np.float64))
    for nfe, lx in zip(nfes, logs):
        x = rect[0] + (lx - logs.min()) / (logs.max() - logs.min()) * (rect[2] - rect[0])
        draw.line((x, rect[1], x, rect[3]), fill=(238, 240, 243, 255), width=1)
        label = str(nfe)
        draw.text((x - 11 * SCALE, rect[3] + 12 * SCALE), label, fill=MUTED, font=font(10))
    draw.rectangle(rect, outline=(88, 94, 104, 255), width=1)

    for values, stds, color in [(diff_values, diff_std, DIFFUSION), (flow_values, flow_std, FLOW)]:
        pts = line_points(nfes, values, rect, y_min, y_max)
        upper = line_points(nfes, [v + s for v, s in zip(values, stds)], rect, y_min, y_max)
        lower = line_points(nfes, [max(0.0, v - s) for v, s in zip(values, stds)], rect, y_min, y_max)
        band = upper + list(reversed(lower))
        draw.polygon(band, fill=(color[0], color[1], color[2], 34))
        for a, b in zip(pts[:-1], pts[1:]):
            draw.line((a[0], a[1], b[0], b[1]), fill=color, width=5 * SCALE)
        for x, y in pts:
            r = 6 * SCALE
            draw.ellipse((x - r, y - r, x + r, y + r), fill=color, outline=(255, 255, 255, 230), width=1 * SCALE)


def save_nfe_curve_slide(metrics: dict, out_path: Path) -> None:
    nfes = metrics["nfe_values"]
    width, height = 1280 * SCALE, 610 * SCALE
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img, "RGBA")

    rect1 = (76 * SCALE, 112 * SCALE, 602 * SCALE, 512 * SCALE)
    rect2 = (704 * SCALE, 112 * SCALE, 1230 * SCALE, 512 * SCALE)

    diff_swd = [metrics["diffusion"][str(n)]["swd"]["mean"] for n in nfes]
    flow_swd = [metrics["flow_matching"][str(n)]["swd"]["mean"] for n in nfes]
    diff_swd_std = [metrics["diffusion"][str(n)]["swd"]["std"] for n in nfes]
    flow_swd_std = [metrics["flow_matching"][str(n)]["swd"]["std"] for n in nfes]
    draw_metric_curve_slide(draw, rect1, nfes, diff_swd, flow_swd, diff_swd_std, flow_swd_std, "分布距离", "fixed-projection SWD, lower is better")

    diff_nn = [metrics["diffusion"][str(n)]["nearest_p90"]["mean"] for n in nfes]
    flow_nn = [metrics["flow_matching"][str(n)]["nearest_p90"]["mean"] for n in nfes]
    diff_nn_std = [metrics["diffusion"][str(n)]["nearest_p90"]["std"] for n in nfes]
    flow_nn_std = [metrics["flow_matching"][str(n)]["nearest_p90"]["std"] for n in nfes]
    draw_metric_curve_slide(draw, rect2, nfes, diff_nn, flow_nn, diff_nn_std, flow_nn_std, "尾部贴近", "p90 nearest-target distance, lower is better")

    legend_y = 22 * SCALE
    draw.rectangle((76 * SCALE, legend_y, 94 * SCALE, legend_y + 18 * SCALE), fill=DIFFUSION)
    draw.text((104 * SCALE, legend_y - 2 * SCALE), "Diffusion / deterministic DDIM-like", fill=INK, font=font(10))
    draw.rectangle((360 * SCALE, legend_y, 378 * SCALE, legend_y + 18 * SCALE), fill=FLOW)
    draw.text((388 * SCALE, legend_y - 2 * SCALE), "Flow Matching / Euler ODE", fill=INK, font=font(10))
    draw.text((rect1[2] - 22 * SCALE, rect1[3] + 46 * SCALE), "NFE", fill=MUTED, font=font(10))
    draw.text((rect2[2] - 22 * SCALE, rect2[3] + 46 * SCALE), "NFE", fill=MUTED, font=font(10))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def draw_scatter_panel(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    target: Array,
    samples: Array,
    color: tuple[int, int, int, int],
    title: str,
) -> None:
    draw.text((rect[0], rect[1] - 25 * SCALE), title, fill=INK, font=font(10))
    for frac in np.linspace(0.0, 1.0, 5):
        x = rect[0] + frac * (rect[2] - rect[0])
        y = rect[1] + frac * (rect[3] - rect[1])
        draw.line((x, rect[1], x, rect[3]), fill=GRID, width=1)
        draw.line((rect[0], y, rect[2], y), fill=GRID, width=1)
    zero = to_pixel(np.array([[0.0, 0.0]]), BOUNDS, rect)[0]
    draw.line((zero[0], rect[1], zero[0], rect[3]), fill=AXIS, width=1)
    draw.line((rect[0], zero[1], rect[2], zero[1]), fill=AXIS, width=1)
    draw.rectangle(rect, outline=(88, 94, 104, 255), width=1)
    for points, fill, radius, limit in [(target, TARGET, 1, 900), (samples, color, 1, 900)]:
        subset = points[:limit] if len(points) > limit else points
        pix = to_pixel(subset, BOUNDS, rect)
        r = max(1, radius * SCALE)
        for px, py in pix:
            if px < rect[0] or px > rect[2] or py < rect[1] or py > rect[3]:
                continue
            draw.ellipse((px - r, py - r, px + r, py + r), fill=fill)


def save_nfe_snapshots(target: Array, snapshots: dict[tuple[str, int], Array], nfes: list[int], out_path: Path) -> None:
    shown_nfes = [n for n in [3, 5, 10, 20, 40, 80] if n in nfes]
    width, height = 1560 * SCALE, 630 * SCALE
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img, "RGBA")
    draw.text((30 * SCALE, 24 * SCALE), "Final samples under different NFE", fill=INK, font=font(22, serif=True))
    draw.text((30 * SCALE, 54 * SCALE), "orange = target S-curve, blue/green = generated samples", fill=MUTED, font=font(10))

    cols = len(shown_nfes)
    cell_w = (width - 130 * SCALE) // cols
    row_h = 245 * SCALE
    panel_size = min(cell_w - 22 * SCALE, row_h - 42 * SCALE)
    top0 = 122 * SCALE
    left0 = 86 * SCALE
    row_names = [("Diffusion", "diffusion", DIFFUSION), ("Flow", "flow_matching", FLOW)]
    for r, (row_label, method, color) in enumerate(row_names):
        row_top = top0 + r * row_h
        draw.text((28 * SCALE, row_top + 74 * SCALE), row_label, fill=INK, font=font(15, serif=True))
        for c, nfe in enumerate(shown_nfes):
            left = left0 + c * cell_w
            rect = (left, row_top + 34 * SCALE, left + panel_size, row_top + 34 * SCALE + panel_size)
            title = f"NFE {nfe}"
            draw_scatter_panel(draw, rect, target, snapshots[(method, nfe)], color, title)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def save_nfe_snapshots_focus(target: Array, snapshots: dict[tuple[str, int], Array], nfes: list[int], out_path: Path) -> None:
    shown_nfes = [n for n in [3, 10, 80] if n in nfes]
    width, height = 760 * SCALE, 560 * SCALE
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img, "RGBA")
    draw.text((30 * SCALE, 24 * SCALE), "Representative final samples", fill=INK, font=font(20, serif=True))
    draw.text((30 * SCALE, 54 * SCALE), "same initialization; orange = target S-curve", fill=MUTED, font=font(10))

    panel_size = 174 * SCALE
    left0 = 118 * SCALE
    top0 = 126 * SCALE
    col_gap = 36 * SCALE
    row_gap = 210 * SCALE
    row_names = [("Diffusion", "diffusion", DIFFUSION), ("Flow", "flow_matching", FLOW)]
    for r, (row_label, method, color) in enumerate(row_names):
        row_top = top0 + r * row_gap
        draw.text((30 * SCALE, row_top + 82 * SCALE), row_label, fill=INK, font=font(15, serif=True))
        for c, nfe in enumerate(shown_nfes):
            left = left0 + c * (panel_size + col_gap)
            rect = (left, row_top + 34 * SCALE, left + panel_size, row_top + 34 * SCALE + panel_size)
            draw_scatter_panel(draw, rect, target, snapshots[(method, nfe)], color, f"NFE {nfe}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def save_nfe_snapshots_focus_slide(target: Array, snapshots: dict[tuple[str, int], Array], nfes: list[int], out_path: Path) -> None:
    shown_nfes = [n for n in [3, 10, 80] if n in nfes]
    width, height = 760 * SCALE, 560 * SCALE
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img, "RGBA")

    panel_size = 194 * SCALE
    left0 = 104 * SCALE
    top0 = 40 * SCALE
    col_gap = 36 * SCALE
    row_gap = 260 * SCALE
    row_names = [("Diffusion", "diffusion", DIFFUSION), ("Flow", "flow_matching", FLOW)]
    for r, (row_label, method, color) in enumerate(row_names):
        row_top = top0 + r * row_gap
        draw.text((24 * SCALE, row_top + 112 * SCALE), row_label, fill=INK, font=font(15, serif=True))
        for c, nfe in enumerate(shown_nfes):
            left = left0 + c * (panel_size + col_gap)
            rect = (left, row_top + 34 * SCALE, left + panel_size, row_top + 34 * SCALE + panel_size)
            draw_scatter_panel(draw, rect, target, snapshots[(method, nfe)], color, f"NFE {nfe}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def main() -> None:
    out_dir = ROOT / "outputs" / "analysis_figures"
    asset_dir = ROOT / "presentation_v2" / "assets"
    model_dir = ROOT / "outputs" / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    asset_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    cfg = ExperimentConfig(seed=7, train_steps=2200, batch_size=512, sample_count=2500, sample_steps=80, frames=28)
    nfes = [3, 5, 10, 20, 40, 80]
    eval_seeds = [7001, 7002, 7003, 7004, 7005]
    model_path = model_dir / "scurve_seed7_train2200_hidden64.npz"
    target_path = model_dir / "scurve_target_seed9001_n2500.npz"

    diffusion_model, flow_model, training_metrics, model_source = get_or_train_models(cfg, model_path)
    target, target_source = get_or_create_target(cfg, target_path, target_seed=9001)
    print(f"models: {model_source} from {model_path}")
    print(f"target: {target_source} from {target_path}")

    metrics, snapshots = evaluate_nfe(diffusion_model, flow_model, cfg, target, nfes, eval_seeds)
    metrics["training"] = training_metrics
    metrics["fixed_assets"] = {
        "model_path": str(model_path.relative_to(ROOT)),
        "model_source": model_source,
        "target_path": str(target_path.relative_to(ROOT)),
        "target_source": target_source,
    }

    metrics_path = out_dir / "11_nfe_comparison_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    save_nfe_curve(metrics, out_dir / "11_nfe_comparison_curve.png")
    save_nfe_curve_slide(metrics, out_dir / "11_nfe_comparison_curve_slide.png")
    save_nfe_snapshots(target, snapshots, nfes, out_dir / "11_nfe_comparison_snapshots.png")
    save_nfe_snapshots_focus(target, snapshots, nfes, out_dir / "11_nfe_comparison_snapshots_focus.png")
    save_nfe_snapshots_focus_slide(target, snapshots, nfes, out_dir / "11_nfe_comparison_snapshots_focus_slide.png")

    for name in [
        "11_nfe_comparison_curve.png",
        "11_nfe_comparison_curve_slide.png",
        "11_nfe_comparison_snapshots.png",
        "11_nfe_comparison_snapshots_focus.png",
        "11_nfe_comparison_snapshots_focus_slide.png",
    ]:
        (asset_dir / name).write_bytes((out_dir / name).read_bytes())

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"saved NFE comparison figures to {out_dir}")


if __name__ == "__main__":
    main()
