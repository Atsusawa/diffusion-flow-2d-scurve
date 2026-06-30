from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "presentation_v2" / "assets"
BOUNDS = (-2.65, 2.65, -2.45, 2.45)

TARGET = (226, 126, 42, 175)
DIFFUSION = (36, 99, 184, 170)
FLOW = (35, 136, 86, 180)
GAUSS = (43, 53, 70, 100)
GRID = (226, 229, 234, 255)
AXIS = (186, 192, 202, 255)
INK = (35, 42, 52, 255)


def to_pixel(points: np.ndarray, rect: tuple[int, int, int, int]) -> np.ndarray:
    xmin, xmax, ymin, ymax = BOUNDS
    left, top, right, bottom = rect
    x = left + (points[:, 0] - xmin) / (xmax - xmin) * (right - left)
    y = bottom - (points[:, 1] - ymin) / (ymax - ymin) * (bottom - top)
    return np.stack([x, y], axis=1)


def draw_base(draw: ImageDraw.ImageDraw, rect: tuple[int, int, int, int]) -> None:
    for frac in np.linspace(0.0, 1.0, 7):
        x = rect[0] + frac * (rect[2] - rect[0])
        y = rect[1] + frac * (rect[3] - rect[1])
        draw.line((x, rect[1], x, rect[3]), fill=GRID, width=2)
        draw.line((rect[0], y, rect[2], y), fill=GRID, width=2)
    zero = to_pixel(np.array([[0.0, 0.0]], dtype=np.float64), rect)[0]
    draw.line((zero[0], rect[1], zero[0], rect[3]), fill=AXIS, width=2)
    draw.line((rect[0], zero[1], rect[2], zero[1]), fill=AXIS, width=2)
    draw.rectangle(rect, outline=(105, 111, 122, 255), width=3)


def draw_points(draw: ImageDraw.ImageDraw, points: np.ndarray, rect: tuple[int, int, int, int], color: tuple[int, int, int, int], radius: int = 3, limit: int = 2500) -> None:
    pts = points[:limit]
    pixels = to_pixel(pts, rect)
    for x, y in pixels:
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)


def draw_arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], color: tuple[int, int, int, int]) -> None:
    draw.line((*start, *end), fill=color, width=7)
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    n = max((dx * dx + dy * dy) ** 0.5, 1.0)
    ux, uy = dx / n, dy / n
    px, py = -uy, ux
    size = 24
    a = (end[0] - size * ux + 0.45 * size * px, end[1] - size * uy + 0.45 * size * py)
    b = (end[0] - size * ux - 0.45 * size * px, end[1] - size * uy - 0.45 * size * py)
    draw.polygon([end, a, b], fill=color)


def save_triptych(samples: dict[str, np.ndarray]) -> None:
    width, height = 3600, 1200
    gap = 72
    panel_w = (width - 2 * gap - 160) // 3
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img, "RGBA")
    panels = [
        samples["target"],
        samples["diffusion"],
        samples["flow_matching"],
    ]
    colors = [TARGET, DIFFUSION, FLOW]
    for i, (points, color) in enumerate(zip(panels, colors)):
        left = 80 + i * (panel_w + gap)
        rect = (left, 90, left + panel_w, height - 90)
        draw_base(draw, rect)
        if i > 0:
            draw_points(draw, samples["target"], rect, (226, 126, 42, 74), radius=3)
        draw_points(draw, points, rect, color, radius=3)
    img.save(OUT / "final_distribution_triptych.png")


def save_transport(samples: dict[str, np.ndarray]) -> None:
    width, height = 3000, 1100
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img, "RGBA")
    rng = np.random.default_rng(99)
    gaussian = rng.normal(size=(2500, 2))
    left = (120, 120, 1130, height - 120)
    right = (1870, 120, 2880, height - 120)
    draw_base(draw, left)
    draw_base(draw, right)
    draw_points(draw, gaussian, left, GAUSS, radius=3)
    draw_points(draw, samples["target"], right, TARGET, radius=3)
    for yoff, alpha in [(-170, 110), (-60, 150), (90, 130), (205, 90)]:
        draw_arrow(draw, (1190, height // 2 + yoff), (1810, height // 2 + int(yoff * 0.45)), (35, 42, 52, alpha))
    img.save(OUT / "transport_overview.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = np.load(ROOT / "outputs" / "samples.npz")
    samples = {key: data[key] for key in data.files}
    save_triptych(samples)
    save_transport(samples)
    print(f"saved deck assets to {OUT}")


if __name__ == "__main__":
    main()
