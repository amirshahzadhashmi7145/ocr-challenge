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
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

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
4. If the sign has no words, only a number, output the number alone. Never invent words or units such as MPH. If SPEED LIMIT (or other words) are printed on the sign, include them.
5. No labels, no quotes, no explanation, no extra punctuation. Just the characters.
6. A vanity plate can spell real words (for example DUMB APE); still output only the registration, never the slogan or state name printed above or below it.
7. Mainland Chinese plates never contain the letters I or O; read those as the digits 1 and 0."""


# --------------------------------------------------------------------------- text utils

def norm(s: str) -> str:
    """The grader's normalisation: uppercase, drop whitespace and the characters - . · _"""
    return re.sub(r"[\s\-\.\u00b7_]", "", (s or "").upper())


_US_STATES = [
    "ALABAMA","ALASKA","ARIZONA","ARKANSAS","CALIFORNIA","COLORADO","CONNECTICUT","DELAWARE","FLORIDA","GEORGIA",
    "HAWAII","IDAHO","ILLINOIS","INDIANA","IOWA","KANSAS","KENTUCKY","LOUISIANA","MAINE","MARYLAND","MASSACHUSETTS",
    "MICHIGAN","MINNESOTA","MISSISSIPPI","MISSOURI","MONTANA","NEBRASKA","NEVADA","NEW HAMPSHIRE","NEW JERSEY",
    "NEW MEXICO","NEW YORK","NORTH CAROLINA","NORTH DAKOTA","OHIO","OKLAHOMA","OREGON","PENNSYLVANIA","RHODE ISLAND",
    "SOUTH CAROLINA","SOUTH DAKOTA","TENNESSEE","TEXAS","UTAH","VERMONT","VIRGINIA","WASHINGTON","WEST VIRGINIA",
    "WISCONSIN","WYOMING","DISTRICT OF COLUMBIA","WASHINGTON DC","WASHINGTON D.C.",
]
_US_SLOGANS = [
    "THE EMPIRE STATE","EMPIRE STATE","EXCELSIOR","THE LONE STAR STATE","LONE STAR STATE","SUNSHINE STATE",
    "MYFLORIDA.COM","IN GOD WE TRUST","GARDEN STATE","LAND OF LINCOLN","FIRST IN FLIGHT","FIRST IN FREEDOM",
    "LIVE FREE OR DIE","FAMOUS POTATOES","SCENIC IDAHO","WILD WONDERFUL","MOUNTAINEERS ARE ALWAYS FREE","ALMOST HEAVEN",
    "SPORTSMAN'S PARADISE","PELICAN STATE","VACATIONLAND","AMERICA'S DAIRYLAND","GREAT LAKES STATE","GREAT LAKES",
    "PURE MICHIGAN","WATER WONDERLAND","GRAND CANYON STATE","BIRTHPLACE OF AVIATION","HOME MEANS NEVADA",
    "THE SILVER STATE","EVERGREEN STATE","SHOW ME STATE","SHOW-ME STATE","BLUEGRASS STATE","UNBRIDLED SPIRIT",
    "HOOSIER STATE","TREASURE STATE","BIG SKY","OCEAN STATE","CONSTITUTION STATE","GREEN MOUNTAIN STATE",
    "THE NATURAL STATE","NATURAL STATE","LAND OF OPPORTUNITY","PEACH STATE","GEORGIA ON MY MIND","VOLUNTEER STATE",
    "MAGNOLIA STATE","YELLOWHAMMER STATE","HEART OF DIXIE","SWEET HOME ALABAMA","SOONER STATE","NATIVE AMERICA",
    "LAND OF ENCHANTMENT","CENTENNIAL STATE","EQUALITY STATE","FOREVER WEST","GEM STATE","BEEHIVE STATE",
    "LIFE ELEVATED","GREATEST SNOW ON EARTH","SPIRIT OF AMERICA","SMILING FACES BEAUTIFUL PLACES",
    "WHILE I BREATHE I HOPE","TAXATION WITHOUT REPRESENTATION","END TAXATION WITHOUT REPRESENTATION","ALOHA STATE",
    "THE LAST FRONTIER","GOLDEN STATE","DMV.CA.GOV","EXPLORE MINNESOTA","LAND OF 10,000 LAKES","10,000 LAKES",
    "10000 LAKES","VIRGINIA IS FOR LOVERS","MARYLAND PROUD","KEYSTONE STATE","VISITPA.COM","THE FIRST STATE",
    "FIRST STATE","LEGENDARY","PEACE GARDEN STATE","DISCOVER THE SPIRIT","THE GOOD LIFE","GREAT FACES GREAT PLACES",
    "PACIFIC WONDERLAND",
]
_BANNER_RE = re.compile(
    r"(?<![A-Z0-9])(?:" + "|".join(re.escape(p) for p in sorted(_US_STATES + _US_SLOGANS, key=len, reverse=True)) + r")(?![A-Z0-9])",
    re.I,
)
_URL_RE = re.compile(r"(?<![A-Z0-9])(?:WWW\.)?[A-Z0-9-]+(?:\.[A-Z0-9-]+)*\.(?:COM|GOV|ORG|NET|US)(?![A-Z0-9])", re.I)


def strip_us_banners(t: str) -> str:
    """Drop US state names, mottos and web addresses printed around a plate number.
    Only applied when something is left afterwards, so a plate that IS such a word survives."""
    s = _URL_RE.sub(" ", t)
    s = _BANNER_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip(" -.,:;'\"")
    return s if len(norm(s)) >= 2 else t


_CN_PLATE_RE = re.compile(r"^([\u4e00-\u9fff])\s*([A-Z])[\s·\-\.]*([A-Z0-9][A-Z0-9\s·\-\.]{3,8})$")


def fix_cn_plate(t: str) -> str:
    """Mainland Chinese plates never use the letters I or O in the serial (GA 36 standard):
    after the province character and letter, I is 1 and O is 0."""
    m = _CN_PLATE_RE.match(t.strip().upper())
    if not m:
        return t
    prov, letter, serial = m.groups()
    serial = re.sub(r"[\s·\-\.]", "", serial).replace("I", "1").replace("O", "0")
    return f"{prov}{letter}·{serial}"


def _noise_level(im: Image.Image) -> float:
    """Mean absolute difference between the grayscale image and its 3x3 median: high for sensor noise."""
    g = im.convert("L")
    a = np.asarray(g, dtype=np.float32)
    m = np.asarray(g.filter(ImageFilter.MedianFilter(3)), dtype=np.float32)
    return float(np.abs(a - m).mean())


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


NOISE_THRESHOLD = 9.0


def variants(im: Image.Image, n: int) -> list:
    """Test-time augmentation chosen by image condition: noisy images get median-filtered variants,
    dark images get brightened, everything else gets contrast plus sharpening."""
    out = [im]
    if n < 2:
        return out
    noisy = _noise_level(im) > NOISE_THRESHOLD
    dark = float(np.asarray(im.convert("L")).mean()) < 80
    base = im.filter(ImageFilter.MedianFilter(3)) if noisy else im
    if dark:
        base = ImageEnhance.Brightness(base).enhance(1.6)
    out.append(ImageOps.autocontrast(base, cutoff=1))
    if n >= 3:
        if noisy:
            out.append(ImageOps.autocontrast(im.filter(ImageFilter.MedianFilter(5)), cutoff=1))
        else:
            out.append(ImageEnhance.Sharpness(ImageOps.autocontrast(base, cutoff=1)).enhance(2.0))
    return out[:n]


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
        raw = self.processor.batch_decode([gen], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
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
            cands.append((fix_cn_plate(strip_us_banners(clean_text(raw))), p, raw))

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
