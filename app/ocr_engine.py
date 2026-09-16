"""OCR engine shared by server.py, app.py and scripts/test_local.py.

Loads one open vision-language model (default Qwen2.5-VL-3B-Instruct), turns an image
into the characters printed on the plate / sign, and returns (text, confidence, info).

Environment variables (all optional):
  MODEL_ID            Hugging Face repo id            default Qwen/Qwen2.5-VL-3B-Instruct
  MODEL_DIR           local weights folder            default /models/<repo name>; used if it exists
  OCR_TTA_PASSES      1..3 image variants voted       default 3  (set 1 on a slow GPU)
  OCR_TIME_BUDGET_S   stop adding passes after this   default 20 (harness limit is 30 s per image)
  OCR_MAX_NEW_TOKENS  generation cap                  default 32
  OCR_DTYPE           bf16 | fp16 | fp32              default bf16 on GPU, fp32 on CPU
  OCR_MAX_PIXELS      cap on image pixels fed to the model   default 1280*28*28 (~1 MP)
  OCR_QUANT           "4bit" = bitsandbytes nf4, DEV ONLY for a small NVIDIA card; never set in the image
"""
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont, ImageOps

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"


def log(msg: str) -> None:
    print(f"[ocr] {msg}", file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- prompt

SYSTEM_PROMPT = (
    "You are a precise OCR engine for vehicle license plates and road signs. "
    "You output only the characters printed on the plate or sign, never anything else."
)

USER_PROMPT = """Read the text in this image and reply with ONLY the characters, on one line.

Rules:
1. License plate: output only the registration number. Do NOT include the state, province or country name, slogans, mottos, county names, web addresses, dealer frames or stickers printed around it (drop things like "CALIFORNIA", "Empire State", "THE LONE STAR STATE", "dmv.ca.gov").
2. Chinese plate: the leading province character and letter ARE part of the registration. Keep them (for example 京A12345 or 沪B88888).
3. Road sign: output the words and numbers printed on the sign in reading order, top to bottom. Join separate lines with a single space (for example STOP, SPEED LIMIT 65, ROAD WORK AHEAD).
4. If the sign shows only a number, output only the number (for example 35). Never add units such as MPH.
5. No labels, no quotes, no explanation, no extra punctuation. Just the characters."""


# --------------------------------------------------------------------------- text utils

def norm(s: str) -> str:
    """The grader's normalisation: uppercase, drop whitespace and the characters - . · _"""
    return re.sub(r"[\s\-\.\u00b7_]", "", (s or "").upper())


_LABEL_RE = re.compile(
    r"^(?:the\s+)?(?:text|answer|output|result|ocr|plate|license\s*plate|registration(?:\s*number)?"
    r"|number|sign|transcription)\s*(?:is|:|reads?|says?)\s*[:\-]?\s*",
    re.I,
)


def _looks_like_prose(line: str) -> bool:
    words = line.split()
    return (len(words) >= 4 and re.search(r"[a-z]", line) is not None) or (
        len(words) >= 3 and line.rstrip().endswith((".", "!"))
    )


def clean_text(raw: str) -> str:
    """Strip labels, quotes, markdown and explanations; join sign lines with single spaces."""
    lines = [l.strip() for l in (raw or "").strip().splitlines() if l.strip()]
    if not lines:
        return ""
    # A second line that reads like a sentence is the model explaining itself: keep only the answer line.
    if len(lines) > 1 and any(_looks_like_prose(l) for l in lines[1:]):
        lines = lines[:1]
    t = " ".join(lines)
    t = t.replace("**", "").replace("`", "")
    t = _LABEL_RE.sub("", t)
    t = t.strip(" \"'“”‘’.:;,。")
    t = re.sub(r"\s+", " ", t)
    return t


# --------------------------------------------------------------------------- image utils

def load_image(path: str, max_side: int = 1280, min_side: int = 640) -> Image.Image:
    """PNG / JPEG / TIFF -> RGB PIL image with a bounded size. Handles EXIF rotation,
    alpha, 16-bit and multi-page TIFF."""
    im = Image.open(path)
    try:
        im = ImageOps.exif_transpose(im)
    except Exception:
        pass
    if getattr(im, "n_frames", 1) > 1:
        im.seek(0)
    if im.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
        a = np.asarray(im, dtype=np.float32)
        lo, hi = np.percentile(a, [0.5, 99.5])
        a = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1) * 255
        im = Image.fromarray(a.astype(np.uint8))
    im = im.convert("RGB")
    w, h = im.size
    long_side = max(w, h)
    if long_side > max_side:
        s = max_side / long_side
    elif long_side < min_side:
        s = min_side / long_side
    else:
        s = 1.0
    if s != 1.0:
        im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
    return im


def variants(im: Image.Image, n: int) -> list:
    """Test-time augmentation: original, contrast-stretched, and a sharpened (brightened if dark) copy."""
    out = [im]
    if n >= 2:
        out.append(ImageOps.autocontrast(im, cutoff=1))
    if n >= 3:
        lum = float(np.asarray(im.convert("L")).mean())
        v = ImageEnhance.Brightness(im).enhance(1.6) if lum < 80 else im
        out.append(ImageEnhance.Sharpness(v).enhance(2.0))
    return out[: max(1, n)]


# --------------------------------------------------------------------------- engine

class OCREngine:
    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        model_dir: str | None = None,
        tta_passes: int = 3,
        time_budget_s: float = 20.0,
        max_new_tokens: int = 32,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1280 * 28 * 28,
    ):
        self.model_id = model_id
        self.model_dir = model_dir or f"/models/{model_id.split('/')[-1]}"
        self.tta_passes = max(1, min(3, tta_passes))
        self.time_budget_s = time_budget_s
        self.max_new_tokens = max_new_tokens
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.model = None
        self.processor = None
        self.device = "cpu"
        self._eos_ids: set = set()

    @classmethod
    def from_env(cls) -> "OCREngine":
        return cls(
            model_id=os.environ.get("MODEL_ID", DEFAULT_MODEL_ID),
            model_dir=os.environ.get("MODEL_DIR") or None,
            tta_passes=int(os.environ.get("OCR_TTA_PASSES", "3")),
            time_budget_s=float(os.environ.get("OCR_TIME_BUDGET_S", "20")),
            max_new_tokens=int(os.environ.get("OCR_MAX_NEW_TOKENS", "32")),
            max_pixels=int(os.environ.get("OCR_MAX_PIXELS", str(1280 * 28 * 28))),
        )

    # ---- loading

    def _source(self) -> tuple[str, bool]:
        d = Path(self.model_dir)
        if d.is_dir() and any(d.glob("*.safetensors")):
            return str(d), True
        return self.model_id, False

    def load(self) -> None:
        import torch
        from transformers import AutoProcessor

        try:
            from transformers import AutoModelForImageTextToText as AutoVLM
        except ImportError:  # older transformers
            from transformers import AutoModelForVision2Seq as AutoVLM

        src, local = self._source()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # OCR_DTYPE overrides for local dev: fp16 for pre-Ampere NVIDIA cards, bf16 to halve RAM on CPU.
        want = os.environ.get("OCR_DTYPE", "bf16" if self.device == "cuda" else "fp32").lower()
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}.get(want, torch.bfloat16)
        t0 = time.time()
        log(f"loading {src} (local={local}) on {self.device} as {dtype}")

        try:
            self.processor = AutoProcessor.from_pretrained(
                src, min_pixels=self.min_pixels, max_pixels=self.max_pixels, local_files_only=local
            )
        except TypeError:
            self.processor = AutoProcessor.from_pretrained(src, local_files_only=local)

        kwargs: dict = {"local_files_only": local}
        quant = os.environ.get("OCR_QUANT", "").lower()  # DEV ONLY (small NVIDIA card): "4bit" via bitsandbytes
        if quant == "4bit":
            try:
                from transformers import BitsAndBytesConfig
                import bitsandbytes  # noqa: F401
            except ImportError as e:
                raise RuntimeError("OCR_QUANT=4bit needs `pip install bitsandbytes` (local dev only, never in the image)") from e
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype
            )
            kwargs["device_map"] = {"": 0}
            log("4-bit quantised load (dev only; the submission runs bf16)")
        try:  # transformers >= 4.56 uses `dtype`, older versions `torch_dtype`
            self.model = AutoVLM.from_pretrained(src, dtype=dtype, **kwargs)
        except TypeError:
            self.model = AutoVLM.from_pretrained(src, torch_dtype=dtype, **kwargs)
        if quant != "4bit":
            self.model.to(self.device)
        self.model.eval()

        eos = self.model.generation_config.eos_token_id
        self._eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos]) - {None}
        if self.model.generation_config.pad_token_id is not None:
            self._eos_ids.add(self.model.generation_config.pad_token_id)

        if self.device == "cuda":
            log(f"gpu: {torch.cuda.get_device_name(0)}; loaded in {time.time() - t0:.1f}s; "
                f"vram {torch.cuda.memory_allocated() / 2**30:.1f} GiB")
        else:
            log(f"loaded on CPU in {time.time() - t0:.1f}s (slow; fine for plumbing tests only)")

    def warmup(self) -> None:
        """One throwaway pass so kernel compilation is not paid on the first graded image."""
        im = Image.new("RGB", (512, 256), "white")
        d = ImageDraw.Draw(im)
        try:
            f = ImageFont.load_default(size=120)
        except TypeError:
            f = ImageFont.load_default()
        d.text((60, 60), "STOP", fill="black", font=f)
        t0 = time.time()
        raw, _ = self._generate(im)
        log(f"warmup {time.time() - t0:.1f}s -> {raw!r}")

    # ---- inference

    def _generate(self, im: Image.Image) -> tuple[str, float]:
        import torch

        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": USER_PROMPT}]},
        ]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[prompt], images=[im], return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                num_beams=1,
                return_dict_in_generate=True,
                output_scores=True,
            )
        gen = out.sequences[0, inputs["input_ids"].shape[1]:]
        raw = self.processor.batch_decode([gen], skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
        return raw, self._mean_token_prob(out.scores, gen)

    def _mean_token_prob(self, scores, gen_ids) -> float:
        try:
            import torch

            probs = []
            for step, tok in zip(scores, gen_ids.tolist()):
                if tok in self._eos_ids:
                    break
                probs.append(torch.softmax(step[0].float(), dim=-1)[tok].item())
            return float(np.mean(probs)) if probs else 0.0
        except Exception:
            return 0.5

    def read(self, image_path: str) -> tuple[str, float, dict]:
        """Returns (text, confidence, info). Runs up to tta_passes variants inside the time budget
        and returns the majority answer under the grader's normalisation."""
        t0 = time.time()
        im = load_image(image_path)
        cands: list[tuple[str, float, str]] = []
        for i, v in enumerate(variants(im, self.tta_passes)):
            elapsed = time.time() - t0
            if i > 0:
                per_pass = elapsed / i
                if elapsed + per_pass > self.time_budget_s:
                    break
            raw, p = self._generate(v)
            cands.append((clean_text(raw), p, raw))

        groups: dict[str, dict] = {}
        for txt, p, _raw in cands:
            g = groups.setdefault(norm(txt), {"txt": txt, "votes": 0, "psum": 0.0})
            g["votes"] += 1
            g["psum"] += p
        best = max(groups.values(), key=lambda g: (bool(norm(g["txt"])), g["votes"], g["psum"] / g["votes"]))
        agreement = best["votes"] / len(cands)
        mean_p = best["psum"] / best["votes"]
        conf = max(0.0, min(1.0, mean_p * (0.6 + 0.4 * agreement)))
        info = {
            "passes": len(cands),
            "agreement": round(agreement, 2),
            "elapsed_s": round(time.time() - t0, 2),
            "candidates": [c[0] for c in cands],
            "raw": [c[2] for c in cands],
        }
        return best["txt"], round(conf, 4), info
