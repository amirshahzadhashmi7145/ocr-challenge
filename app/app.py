#!/usr/bin/env python3
"""Harness entry point. The grader runs, inside the already-running container:

    python3 /app/app.py --input-image /app/input/image_01.png

and expects /app/output/image_01_output.json  ->  {"text": "7ABC123", "confidence": 0.94}

Normal path: ask the resident server (server.py) that already holds the model.
Fallback: if no server exists at all (connection refused), load the model in this process,
but never let a single image run past the 30 s budget: an empty answer costs one image,
a timeout kills the whole run.
A JSON file is ALWAYS written, and the exit code is always 0.
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PORT = os.environ.get("OCR_PORT", "8765")
SERVER = os.environ.get("OCR_SERVER", f"http://127.0.0.1:{PORT}")
OUTPUT_DIR = Path(os.environ.get("OCR_OUTPUT_DIR", "/app/output"))
NO_SERVER_GRACE_S = float(os.environ.get("OCR_NO_SERVER_GRACE_S", "8"))    # refused this long -> no server exists
LOADING_WAIT_S = float(os.environ.get("OCR_LOADING_WAIT_S", "600"))         # server up but loading -> wait (startup budget)
REQUEST_TIMEOUT_S = float(os.environ.get("OCR_REQUEST_TIMEOUT_S", "28"))
IN_PROCESS_DEADLINE_S = float(os.environ.get("OCR_IN_PROCESS_DEADLINE_S", "22"))


def log(msg: str) -> None:
    print(f"[app] {msg}", file=sys.stderr, flush=True)


def http_json(method: str, url: str, payload: dict | None = None, timeout: float = 5.0) -> dict:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def wait_for_server() -> str:
    """'ready' | 'absent' (nothing listening) | 'error' (model failed to load) | 'timeout'"""
    t0 = time.time()
    refused_since = None
    while True:
        try:
            http_json("GET", SERVER + "/health", timeout=3)
            return "ready"
        except urllib.error.HTTPError as e:
            refused_since = None
            if e.code == 503:
                if time.time() - t0 > LOADING_WAIT_S:
                    return "timeout"
            else:
                log(f"server reports failure: {e.read()[:300]!r}")
                return "error"
        except Exception:  # connection refused / reset
            refused_since = refused_since or time.time()
            if time.time() - refused_since > NO_SERVER_GRACE_S:
                return "absent"
        time.sleep(0.5)


def read_via_server(path: Path) -> tuple[str, float]:
    res = http_json("POST", SERVER + "/ocr", {"image_path": str(path)}, timeout=REQUEST_TIMEOUT_S)
    return res["text"], float(res.get("confidence", 0.0))


def read_in_process(path: Path, t0: float) -> tuple[str, float]:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ocr_engine import OCREngine

    os.environ.setdefault("OCR_TTA_PASSES", "1")
    eng = OCREngine.from_env()
    eng.time_budget_s = max(2.0, IN_PROCESS_DEADLINE_S - (time.time() - t0) - 2.0)
    eng.load()
    if time.time() - t0 > IN_PROCESS_DEADLINE_S:
        log("model loaded too late for this image; writing empty answer rather than risk the run")
        return "", 0.0
    text, conf, _ = eng.read(str(path))
    return text, conf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-image", required=True)
    ap.add_argument("--output-dir", default=str(OUTPUT_DIR))
    a = ap.parse_args()

    t0 = time.time()
    src = Path(a.input_image)
    out_dir = Path(a.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{src.stem}_output.json"

    text, conf, mode = "", 0.0, "failed"
    try:
        status = wait_for_server()
        if status == "ready":
            text, conf = read_via_server(src)
            mode = "server"
        elif status == "absent":
            log("no server listening; loading the model in-process")
            text, conf = read_in_process(src, t0)
            mode = "in-process"
        else:
            log(f"server status {status}; writing empty answer")
            mode = status
    except Exception as e:
        log(f"FAILED: {type(e).__name__}: {e}")

    conf = max(0.0, min(1.0, float(conf or 0.0)))
    out_path.write_text(json.dumps({"text": text, "confidence": round(conf, 4)}, ensure_ascii=False), encoding="utf-8")
    log(f"{src.name} -> {out_path} [{mode}] text={text!r} conf={conf:.2f} {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
