from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


Array = np.ndarray


@dataclass
class ExperimentConfig:
    seed: int = 7
    train_steps: int = 2200
    batch_size: int = 512
    hidden_size: int = 64
    hidden_layers: int = 3
    learning_rate: float = 2e-3
    sample_count: int = 2500
    sample_steps: int = 80
    frames: int = 28
    diffusion_t_min: float = 0.02
    diffusion_t_max: float = 0.98


def sample_s_curve(n: int, rng: np.random.Generator, noise: float = 0.055) -> Array:
    """Two-dimensional S-shaped target distribution."""
    u = rng.uniform(-2.05, 2.05, size=n)
    x = 1.15 * np.sin(1.65 * u)
    y = 0.82 * u
    points = np.stack([x, y], axis=1)
    points += rng.normal(0.0, noise, size=points.shape)
    return points.astype(np.float64)


def time_features(x: Array, t: Array) -> Array:
    freqs = np.array([1.0, 2.0, 4.0, 8.0], dtype=np.float64)
    angles = 2.0 * math.pi * t * freqs.reshape(1, -1)
    return np.concatenate([x, t, np.sin(angles), np.cos(angles)], axis=1)


def diffusion_alpha_sigma(t: Array | float) -> tuple[Array | float, Array | float]:
    angle = 0.5 * math.pi * t
    return np.cos(angle), np.sin(angle)


class MLP:
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_size: int,
        hidden_layers: int,
        rng: np.random.Generator,
    ) -> None:
        dims = [input_dim] + [hidden_size] * hidden_layers + [output_dim]
        self.weights: list[Array] = []
        self.biases: list[Array] = []
        for fan_in, fan_out in zip(dims[:-1], dims[1:]):
            limit = math.sqrt(6.0 / (fan_in + fan_out))
            self.weights.append(rng.uniform(-limit, limit, size=(fan_in, fan_out)))
            self.biases.append(np.zeros((1, fan_out), dtype=np.float64))

    def parameters(self) -> list[Array]:
        params: list[Array] = []
        for w, b in zip(self.weights, self.biases):
            params.append(w)
            params.append(b)
        return params

    def predict(self, x: Array) -> Array:
        a = x
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ w + b
            if i == len(self.weights) - 1:
                a = z
            else:
                a = np.tanh(z)
        return a

    def loss_and_grads(self, x: Array, target: Array) -> tuple[float, list[Array]]:
        activations = [x]
        a = x
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ w + b
            if i == len(self.weights) - 1:
                a = z
            else:
                a = np.tanh(z)
            activations.append(a)

        pred = activations[-1]
        diff = pred - target
        loss = float(np.mean(diff * diff))
        grad = (2.0 / diff.size) * diff

        d_weights: list[Array] = [np.empty_like(w) for w in self.weights]
        d_biases: list[Array] = [np.empty_like(b) for b in self.biases]

        for layer in reversed(range(len(self.weights))):
            d_weights[layer] = activations[layer].T @ grad
            d_biases[layer] = np.sum(grad, axis=0, keepdims=True)
            if layer > 0:
                grad = (grad @ self.weights[layer].T) * (1.0 - activations[layer] ** 2)

        grads: list[Array] = []
        for dw, db in zip(d_weights, d_biases):
            grads.append(dw)
            grads.append(db)
        return loss, grads


class Adam:
    def __init__(
        self,
        params: list[Array],
        lr: float = 2e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]

    def step(self, grads: list[Array]) -> None:
        self.t += 1
        lr_t = self.lr * math.sqrt(1.0 - self.beta2**self.t) / (1.0 - self.beta1**self.t)
        for i, (p, g) in enumerate(zip(self.params, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * (g * g)
            p -= lr_t * self.m[i] / (np.sqrt(self.v[i]) + self.eps)


def make_model(cfg: ExperimentConfig, rng: np.random.Generator) -> MLP:
    probe_x = np.zeros((1, 2), dtype=np.float64)
    probe_t = np.zeros((1, 1), dtype=np.float64)
    input_dim = time_features(probe_x, probe_t).shape[1]
    return MLP(input_dim, 2, cfg.hidden_size, cfg.hidden_layers, rng)


def train_diffusion(cfg: ExperimentConfig, rng: np.random.Generator) -> tuple[MLP, dict]:
    model = make_model(cfg, rng)
    opt = Adam(model.parameters(), lr=cfg.learning_rate)
    start = time.perf_counter()
    losses: list[float] = []

    for step in range(1, cfg.train_steps + 1):
        data = sample_s_curve(cfg.batch_size, rng)
        eps = rng.normal(size=(cfg.batch_size, 2))
        t = rng.uniform(cfg.diffusion_t_min, cfg.diffusion_t_max, size=(cfg.batch_size, 1))
        alpha, sigma = diffusion_alpha_sigma(t)
        noisy = alpha * data + sigma * eps
        loss, grads = model.loss_and_grads(time_features(noisy, t), eps)
        opt.step(grads)
        losses.append(loss)
        if step == 1 or step % max(1, cfg.train_steps // 5) == 0:
            print(f"diffusion step {step:5d}/{cfg.train_steps}: loss={loss:.5f}")

    return model, {
        "loss_final": losses[-1],
        "loss_mean_last_100": float(np.mean(losses[-100:])),
        "train_seconds": time.perf_counter() - start,
    }


def train_flow_matching(cfg: ExperimentConfig, rng: np.random.Generator) -> tuple[MLP, dict]:
    model = make_model(cfg, rng)
    opt = Adam(model.parameters(), lr=cfg.learning_rate)
    start = time.perf_counter()
    losses: list[float] = []

    for step in range(1, cfg.train_steps + 1):
        x0 = rng.normal(size=(cfg.batch_size, 2))
        x1 = sample_s_curve(cfg.batch_size, rng)
        t = rng.uniform(0.0, 1.0, size=(cfg.batch_size, 1))
        xt = (1.0 - t) * x0 + t * x1
        velocity = x1 - x0
        loss, grads = model.loss_and_grads(time_features(xt, t), velocity)
        opt.step(grads)
        losses.append(loss)
        if step == 1 or step % max(1, cfg.train_steps // 5) == 0:
            print(f"flow       step {step:5d}/{cfg.train_steps}: loss={loss:.5f}")

    return model, {
        "loss_final": losses[-1],
        "loss_mean_last_100": float(np.mean(losses[-100:])),
        "train_seconds": time.perf_counter() - start,
    }


def frame_indices(total_steps: int, frames: int) -> set[int]:
    return set(np.unique(np.linspace(0, total_steps, frames, dtype=int)).tolist())


def sample_diffusion(
    model: MLP,
    cfg: ExperimentConfig,
    rng: np.random.Generator,
) -> tuple[Array, list[tuple[float, Array]], dict]:
    start = time.perf_counter()
    x = rng.normal(size=(cfg.sample_count, 2))
    times = np.linspace(cfg.diffusion_t_max, 0.0, cfg.sample_steps + 1)
    keep = frame_indices(cfg.sample_steps, cfg.frames)
    states: list[tuple[float, Array]] = []
    if 0 in keep:
        states.append((times[0], x.copy()))

    for i in range(cfg.sample_steps):
        t_cur = times[i]
        t_next = times[i + 1]
        t_batch = np.full((cfg.sample_count, 1), t_cur, dtype=np.float64)
        eps_pred = model.predict(time_features(x, t_batch))

        alpha, sigma = diffusion_alpha_sigma(t_cur)
        alpha_next, sigma_next = diffusion_alpha_sigma(t_next)
        x0_hat = (x - sigma * eps_pred) / max(float(alpha), 1e-3)
        x0_hat = np.clip(x0_hat, -4.0, 4.0)
        x = alpha_next * x0_hat + sigma_next * eps_pred

        if i + 1 in keep:
            states.append((t_next, x.copy()))

    return x, states, {"sample_seconds": time.perf_counter() - start, "nfe": cfg.sample_steps}


def sample_flow_matching(
    model: MLP,
    cfg: ExperimentConfig,
    rng: np.random.Generator,
) -> tuple[Array, list[tuple[float, Array]], dict]:
    start = time.perf_counter()
    x = rng.normal(size=(cfg.sample_count, 2))
    times = np.linspace(0.0, 1.0, cfg.sample_steps + 1)
    keep = frame_indices(cfg.sample_steps, cfg.frames)
    states: list[tuple[float, Array]] = []
    if 0 in keep:
        states.append((times[0], x.copy()))

    for i in range(cfg.sample_steps):
        t_cur = times[i]
        dt = times[i + 1] - times[i]
        t_batch = np.full((cfg.sample_count, 1), t_cur, dtype=np.float64)
        velocity = model.predict(time_features(x, t_batch))
        x = x + dt * velocity
        if i + 1 in keep:
            states.append((times[i + 1], x.copy()))

    return x, states, {"sample_seconds": time.perf_counter() - start, "nfe": cfg.sample_steps}


def sliced_wasserstein_distance(a: Array, b: Array, rng: np.random.Generator, projections: int = 128) -> float:
    n = min(len(a), len(b))
    if len(a) != n:
        a = a[rng.choice(len(a), size=n, replace=False)]
    if len(b) != n:
        b = b[rng.choice(len(b), size=n, replace=False)]

    dirs = rng.normal(size=(projections, 2))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12
    distances = []
    for direction in dirs:
        pa = np.sort(a @ direction)
        pb = np.sort(b @ direction)
        distances.append(np.mean(np.abs(pa - pb)))
    return float(np.mean(distances))


def to_pixel(points: Array, bounds: tuple[float, float, float, float], rect: tuple[int, int, int, int]) -> Array:
    xmin, xmax, ymin, ymax = bounds
    left, top, right, bottom = rect
    x = left + (points[:, 0] - xmin) / (xmax - xmin) * (right - left)
    y = bottom - (points[:, 1] - ymin) / (ymax - ymin) * (bottom - top)
    return np.stack([x, y], axis=1)


def draw_points(draw: ImageDraw.ImageDraw, pixels: Array, color: tuple[int, int, int, int], radius: int) -> None:
    for px, py in pixels:
        draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=color)


def render_scatter(
    points: Array,
    title: str,
    target: Array | None = None,
    width: int = 640,
    height: int = 520,
    bounds: tuple[float, float, float, float] = (-2.55, 2.55, -2.35, 2.35),
    accent: tuple[int, int, int, int] = (31, 104, 196, 185),
) -> Image.Image:
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img, "RGBA")
    font = ImageFont.load_default()
    rect = (54, 42, width - 24, height - 44)

    for frac in np.linspace(0.0, 1.0, 7):
        x = rect[0] + frac * (rect[2] - rect[0])
        y = rect[1] + frac * (rect[3] - rect[1])
        draw.line((x, rect[1], x, rect[3]), fill=(225, 228, 232, 255))
        draw.line((rect[0], y, rect[2], y), fill=(225, 228, 232, 255))

    zero_x = to_pixel(np.array([[0.0, 0.0]]), bounds, rect)[0, 0]
    zero_y = to_pixel(np.array([[0.0, 0.0]]), bounds, rect)[0, 1]
    draw.line((zero_x, rect[1], zero_x, rect[3]), fill=(180, 185, 194, 255))
    draw.line((rect[0], zero_y, rect[2], zero_y), fill=(180, 185, 194, 255))
    draw.rectangle(rect, outline=(70, 74, 82, 255), width=1)

    if target is not None:
        target_px = to_pixel(target, bounds, rect)
        draw_points(draw, target_px, (235, 129, 37, 75), radius=1)

    point_px = to_pixel(points, bounds, rect)
    draw_points(draw, point_px, accent, radius=1)
    draw.text((18, 14), title, fill=(32, 35, 40), font=font)
    draw.text((18, height - 24), "orange = target S curve, blue/green = generated particles", fill=(76, 80, 88), font=font)
    return img


def save_gif(states: list[tuple[float, Array]], target: Array, path: Path, title_prefix: str, accent: tuple[int, int, int, int]) -> None:
    frames = [
        render_scatter(points, f"{title_prefix}   t={t_value:0.3f}", target=target, accent=accent)
        for t_value, points in states
    ]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=120, loop=0)


def save_comparison(target: Array, diffusion: Array, flow: Array, path: Path) -> None:
    panels = [
        render_scatter(target, "Target S-curve samples", target=None, width=420, height=390, accent=(235, 129, 37, 175)),
        render_scatter(diffusion, "Diffusion DDIM samples", target=target, width=420, height=390, accent=(31, 104, 196, 185)),
        render_scatter(flow, "Flow matching samples", target=target, width=420, height=390, accent=(30, 142, 82, 185)),
    ]
    canvas = Image.new("RGB", (420 * 3, 390), "white")
    for i, panel in enumerate(panels):
        canvas.paste(panel, (420 * i, 0))
    canvas.save(path)


def run_experiment(cfg: ExperimentConfig, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    master_rng = np.random.default_rng(cfg.seed)
    diffusion_rng = np.random.default_rng(master_rng.integers(0, 2**32 - 1))
    flow_rng = np.random.default_rng(master_rng.integers(0, 2**32 - 1))
    eval_rng = np.random.default_rng(master_rng.integers(0, 2**32 - 1))

    print("Training diffusion model")
    diffusion_model, diffusion_train = train_diffusion(cfg, diffusion_rng)
    print("Training flow matching model")
    flow_model, flow_train = train_flow_matching(cfg, flow_rng)

    target = sample_s_curve(cfg.sample_count, eval_rng)
    diffusion_samples, diffusion_states, diffusion_sample = sample_diffusion(diffusion_model, cfg, eval_rng)
    flow_samples, flow_states, flow_sample = sample_flow_matching(flow_model, cfg, eval_rng)

    metrics = {
        "config": asdict(cfg),
        "diffusion": {
            **diffusion_train,
            **diffusion_sample,
            "sliced_wasserstein": sliced_wasserstein_distance(diffusion_samples, target, eval_rng),
        },
        "flow_matching": {
            **flow_train,
            **flow_sample,
            "sliced_wasserstein": sliced_wasserstein_distance(flow_samples, target, eval_rng),
        },
    }

    np.savez(
        out_dir / "samples.npz",
        target=target,
        diffusion=diffusion_samples,
        flow_matching=flow_samples,
    )
    save_gif(
        diffusion_states,
        target,
        out_dir / "diffusion_process.gif",
        "Diffusion reverse process",
        (31, 104, 196, 185),
    )
    save_gif(
        flow_states,
        target,
        out_dir / "flow_matching_process.gif",
        "Flow matching ODE process",
        (30, 142, 82, 185),
    )
    save_comparison(target, diffusion_samples, flow_samples, out_dir / "comparison.png")

    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="2D S-curve diffusion vs flow matching experiment")
    parser.add_argument("--out", type=Path, default=Path("outputs"), help="Output directory")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--train-steps", type=int, default=2200)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--hidden-layers", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--sample-count", type=int, default=2500)
    parser.add_argument("--sample-steps", type=int, default=80)
    parser.add_argument("--frames", type=int, default=28)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Small run for environment checks; output quality is intentionally lower.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ExperimentConfig(
        seed=args.seed,
        train_steps=args.train_steps,
        batch_size=args.batch_size,
        hidden_size=args.hidden_size,
        hidden_layers=args.hidden_layers,
        learning_rate=args.learning_rate,
        sample_count=args.sample_count,
        sample_steps=args.sample_steps,
        frames=args.frames,
    )
    if args.quick:
        cfg.train_steps = min(cfg.train_steps, 160)
        cfg.batch_size = min(cfg.batch_size, 256)
        cfg.sample_count = min(cfg.sample_count, 800)
        cfg.sample_steps = min(cfg.sample_steps, 32)
        cfg.frames = min(cfg.frames, 12)
    run_experiment(cfg, args.out)


if __name__ == "__main__":
    main()
