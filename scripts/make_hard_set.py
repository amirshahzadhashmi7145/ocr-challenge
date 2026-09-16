#!/usr/bin/env python3
"""Build a harder OCR set from test_images_real/ for notebook comparison.

Each source image gets three degraded copies in test_images_hard/: gaussian blur,
heavy noise, and darkened-plus-glare (helpers from make_test_images.py).

  python3 scripts/make_hard_set.py
"""
import json
import sys
from pathlib import Path

from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_test_images import low_light_glare, noise  # noqa: E402

SRC = Path("test_images_real")
OUT = Path("test_images_hard")


def gaussian_blur(im: Image.Image) -> Image.Image:
    return im.filter(ImageFilter.GaussianBlur(2.8))


def save(im: Image.Image, dest: Path) -> None:
    ext = dest.suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        im.convert("RGB").save(dest, quality=85)
    else:
        im.save(dest)


def main() -> None:
    expected_src = json.loads((SRC / "expected.json").read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    expected: dict[str, str] = {}
    degradations = (
        ("blur", gaussian_blur),
        ("noise", lambda im: noise(im, sigma=60)),
        ("glare", low_light_glare),
    )
    for name, answer in expected_src.items():
        src = SRC / name
        im = Image.open(src).convert("RGB")
        for tag, fn in degradations:
            out_name = f"{src.stem}_{tag}{src.suffix}"
            save(fn(im), OUT / out_name)
            expected[out_name] = answer
            print(f"  {out_name:<44} -> {answer}")
    (OUT / "expected.json").write_text(
        json.dumps(expected, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"\n{len(expected)} images + expected.json written to {OUT}/")


if __name__ == "__main__":
    main()
