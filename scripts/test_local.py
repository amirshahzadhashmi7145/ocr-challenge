#!/usr/bin/env python3
"""Run the OCR engine directly (no server, no Docker) over a folder of images and grade it,
or grade JSON files written by a harness run. Works in the AMD notebook and on a droplet.

  python3 scripts/test_local.py --images test_images                 # predict + time + grade
  python3 scripts/test_local.py --images test_images --passes 1      # speed check for slow GPUs
  python3 scripts/test_local.py --images test_images --model Qwen/Qwen2.5-VL-7B-Instruct
  python3 scripts/test_local.py --images test_images --grade harness_output

Grading uses <images>/expected.json  {"file.png": "ANSWER", ...}  with the challenge's
normalisation (uppercase, drop whitespace and - . · _). Files without an entry are just printed.
MODEL_ID / MODEL_DIR / HF_HOME environment variables are respected.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
from ocr_engine import OCREngine, norm  # noqa: E402

EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in EXTS)


def load_expected(folder: Path) -> dict:
    p = folder / "expected.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def report(rows: list[tuple[str, str, str | None, float]]) -> None:
    graded = [r for r in rows if r[2] is not None]
    passed = sum(1 for _, pred, exp, _ in graded if norm(pred) == norm(exp))
    for name, pred, exp, secs in rows:
        mark = "  " if exp is None else ("OK  " if norm(pred) == norm(exp) else "FAIL")
        exp_s = "" if exp is None else f"   expected {exp!r}"
        print(f"{mark} {name:<34} {secs:5.1f}s   {pred!r}{exp_s}")
    if graded:
        print(f"\n{passed}/{len(graded)} correct  ->  {passed * 20} / {len(graded) * 20} points at 20 per image")
    slow = [r for r in rows if r[3] > 30]
    if slow:
        print(f"WARNING: {len(slow)} image(s) over the 30 s per-image limit")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="folder with test images (+ optional expected.json)")
    ap.add_argument("--grade", help="folder of <name>_output.json written by a harness run; grade instead of predicting")
    ap.add_argument("--model", help="override MODEL_ID")
    ap.add_argument("--passes", type=int, help="override OCR_TTA_PASSES")
    ap.add_argument("--save", help="also write <name>_output.json files here, like the harness would")
    a = ap.parse_args()

    images = Path(a.images)
    expected = load_expected(images)

    if a.grade:
        rows = []
        for img in list_images(images):
            out = Path(a.grade) / f"{img.stem}_output.json"
            pred = json.loads(out.read_text(encoding="utf-8"))["text"] if out.exists() else "<missing>"
            rows.append((img.name, pred, expected.get(img.name), 0.0))
        report(rows)
        return

    if a.model:
        os.environ["MODEL_ID"] = a.model
    if a.passes:
        os.environ["OCR_TTA_PASSES"] = str(a.passes)

    eng = OCREngine.from_env()
    t0 = time.time()
    eng.load()
    eng.warmup()
    print(f"model ready in {time.time() - t0:.1f}s\n")

    rows = []
    for img in list_images(images):
        t1 = time.time()
        try:
            text, conf, info = eng.read(str(img))
        except Exception as e:  # keep going, show the failure
            text, conf, info = f"<error {type(e).__name__}: {e}>", 0.0, {}
        secs = time.time() - t1
        rows.append((img.name, text, expected.get(img.name), secs))
        cands = info.get("candidates")
        if cands and len(set(map(norm, cands))) > 1:
            print(f"     ({img.name}: passes disagreed: {cands})")
        if a.save:
            Path(a.save).mkdir(parents=True, exist_ok=True)
            (Path(a.save) / f"{img.stem}_output.json").write_text(
                json.dumps({"text": text, "confidence": conf}, ensure_ascii=False), encoding="utf-8"
            )
    print()
    report(rows)


if __name__ == "__main__":
    main()
