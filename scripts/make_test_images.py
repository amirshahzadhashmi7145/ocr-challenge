#!/usr/bin/env python3
"""Generate a starter test set that mirrors the challenge categories and adverse conditions,
plus test_images/expected.json. Needs only Pillow + numpy, so it also runs inside the AMD notebook.

  python3 scripts/make_test_images.py            # writes ./test_images
  python3 scripts/make_test_images.py out_dir

Synthetic images are easier than the graded set: add real photos (and the 10 samples cropped
from the challenge PDF, including the Chinese plates) to the same folder and extend expected.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

OUT = Path("test_images")
rng = np.random.default_rng(0)
EXPECTED: dict[str, str] = {}


def font(size: int):
    try:
        return ImageFont.load_default(size=size)  # Pillow >= 10.1 ships a scalable default font
    except TypeError:
        for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "arial.ttf"):
            try:
                return ImageFont.truetype(name, size)
            except OSError:
                continue
        return ImageFont.load_default()


def center_text(d: ImageDraw.ImageDraw, cx: float, cy: float, text: str, size: int, fill, max_w: float | None = None) -> None:
    f = font(size)
    l, t, r, b = d.textbbox((0, 0), text, font=f)
    while max_w and (r - l) > max_w and size > 20:  # shrink until it fits inside the plate/sign
        size -= 6
        f = font(size)
        l, t, r, b = d.textbbox((0, 0), text, font=f)
    d.text((cx - (r - l) / 2 - l, cy - (b - t) / 2 - t), text, font=f, fill=fill)


def plate(number: str, top: str, bottom: str, bg, fg, banner_fg, size=(640, 320)) -> Image.Image:
    im = Image.new("RGB", size, bg)
    d = ImageDraw.Draw(im)
    d.rectangle([4, 4, size[0] - 5, size[1] - 5], outline=(60, 60, 60), width=6)
    center_text(d, size[0] / 2, 46, top, 40, banner_fg)
    center_text(d, size[0] / 2, size[1] / 2 + 8, number, 150, fg, max_w=size[0] - 70)
    center_text(d, size[0] / 2, size[1] - 40, bottom, 28, banner_fg)
    return im


def perspective(im: Image.Image, dx: int = 120, dy: int = 40) -> Image.Image:
    im = ImageOps.expand(im, border=(90, 70), fill=(90, 90, 90))  # margin so no character is clipped
    w, h = im.size
    src = [(0, 0), (w, 0), (w, h), (0, h)]
    dst = [(dx, dy), (w - dx // 3, 0), (w, h), (dx // 2, h - dy)]
    a = []
    for (x, y), (u, v) in zip(dst, src):  # map output -> input
        a.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        a.append([0, 0, 0, x, y, 1, -v * x, -v * y])
    b = np.array(src, dtype=np.float64).reshape(8)
    coeffs = np.linalg.solve(np.array(a, dtype=np.float64), b)
    return im.transform(im.size, Image.PERSPECTIVE, tuple(coeffs), Image.BICUBIC, fillcolor=(90, 90, 90))


def motion_blur(im: Image.Image) -> Image.Image:
    k = [0.0] * 25
    for i in range(5):
        k[2 * 5 + i] = 1.0  # horizontal line through the 5x5 kernel
    im = im.filter(ImageFilter.Kernel((5, 5), k, scale=5))
    im = im.filter(ImageFilter.Kernel((5, 5), k, scale=5))
    return im.filter(ImageFilter.GaussianBlur(1.6))


def noise(im: Image.Image, sigma: float = 45) -> Image.Image:
    a = np.asarray(im).astype(np.float32) + rng.normal(0, sigma, np.asarray(im).shape)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def low_light_glare(im: Image.Image) -> Image.Image:
    im = ImageEnhance.Brightness(im).enhance(0.32)
    glare = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(glare)
    w, h = im.size
    d.ellipse([w * 0.15, -h * 0.4, w * 0.75, h * 0.55], fill=(255, 255, 240, 150))
    glare = glare.filter(ImageFilter.GaussianBlur(35))
    return Image.alpha_composite(im.convert("RGBA"), glare).convert("RGB")


def octagon_stop() -> Image.Image:
    im = Image.new("RGB", (560, 560), (200, 200, 205))
    d = ImageDraw.Draw(im)
    d.regular_polygon((280, 280, 250), 8, rotation=22.5, fill=(200, 20, 30), outline="white", width=10)
    center_text(d, 280, 280, "STOP", 150, "white")
    return im


def speed_limit(num: str) -> Image.Image:
    im = Image.new("RGB", (480, 620), (215, 215, 215))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([30, 30, 450, 590], radius=24, fill="white", outline="black", width=10)
    center_text(d, 240, 120, "SPEED", 76, "black")
    center_text(d, 240, 210, "LIMIT", 76, "black")
    center_text(d, 240, 400, num, 230, "black")
    return im


def diamond(lines: list[str], color=(255, 140, 0)) -> Image.Image:
    im = Image.new("RGB", (680, 680), (120, 150, 190))
    d = ImageDraw.Draw(im)
    d.polygon([(340, 30), (650, 340), (340, 650), (30, 340)], fill=color, outline="black", width=10)
    y = 340 - 45 * (len(lines) - 1)
    for line in lines:
        center_text(d, 340, y, line, 80, "black")
        y += 90
    return im


def plaque(num: str) -> Image.Image:
    im = Image.new("RGB", (520, 380), (150, 150, 150))
    d = ImageDraw.Draw(im)
    d.rectangle([20, 20, 500, 360], fill=(255, 205, 0), outline="black", width=10)
    center_text(d, 260, 190, num, 220, "black")
    return im


def save(im: Image.Image, name: str, answer: str, **kw) -> None:
    im.save(OUT / name, **kw)
    EXPECTED[name] = answer
    print(f"  {name:<28} -> {answer}")


if __name__ == "__main__":
    OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "test_images")
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"writing to {OUT}/")
    ca = plate("7ABC123", "CALIFORNIA", "dmv.ca.gov", "white", (25, 35, 120), (190, 30, 40))
    save(ca, "plate_ca_clean.png", "7ABC123")
    save(ca.resize((200, 100), Image.LANCZOS), "plate_ca_tiny.png", "7ABC123")
    ny = plate("JHT 2951", "NEW YORK", "EMPIRE STATE", (255, 214, 90), (20, 40, 110), (20, 40, 110))
    save(perspective(ny), "plate_ny_angled.jpg", "JHT 2951", quality=85)
    tx = plate("5XYZ891", "TEXAS", "THE LONE STAR STATE", "white", "black", (40, 40, 40))
    save(motion_blur(tx), "plate_tx_motion_blur.png", "5XYZ891")
    nv = plate("8KLM456", "NEVADA", "HOME MEANS NEVADA", (235, 240, 255), (30, 30, 120), (120, 60, 60))
    save(low_light_glare(nv), "plate_nv_lowlight_glare.jpg", "8KLM456", quality=80)
    save(octagon_stop(), "stop_clean.png", "STOP")
    save(noise(octagon_stop()), "stop_noise.tiff", "STOP")
    save(speed_limit("65"), "speed_limit_65.jpg", "SPEED LIMIT 65", quality=88)
    save(perspective(speed_limit("45"), dx=90, dy=60).filter(ImageFilter.GaussianBlur(1.2)), "speed_limit_45_angled_blur.jpg", "SPEED LIMIT 45", quality=80)
    save(diamond(["ROAD", "WORK", "AHEAD"]), "road_work_ahead.png", "ROAD WORK AHEAD")
    save(noise(diamond(["ICY", "BRIDGE"], color=(255, 200, 0)), sigma=30), "icy_bridge_noise.png", "ICY BRIDGE")
    save(plaque("35"), "advisory_35.tiff", "35")

    (OUT / "expected.json").write_text(json.dumps(EXPECTED, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{len(EXPECTED)} images + expected.json written. Add real photos and the PDF samples next.")
