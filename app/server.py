#!/usr/bin/env python3
"""Resident OCR server. Started once by the container ENTRYPOINT and kept alive.

Binds the port immediately so GET /health answers 503 while the model loads (instead of
"connection refused"), loads and warms the model in a background thread, then serves
POST /ocr {"image_path": "..."} -> {"text": "...", "confidence": 0.93, ...} one request at a time.

    GET  /health   200 ready | 503 loading | 500 model failed to load
    POST /ocr      200 result | 503 loading | 500 inference error
"""
import json
import os
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ocr_engine import OCREngine, log  # noqa: E402

HOST = "127.0.0.1"
PORT = int(os.environ.get("OCR_PORT", "8765"))
READY_FILE = "/tmp/ocr_ready"

state = {"engine": None, "ready": False, "error": None, "t0": time.time()}
gpu_lock = threading.Lock()


def load_model() -> None:
    try:
        eng = OCREngine.from_env()
        eng.load()
        eng.warmup()
        state["engine"] = eng
        state["ready"] = True
        with open(READY_FILE, "w") as f:
            f.write(str(time.time()))
        log(f"READY after {time.time() - state['t0']:.1f}s")
    except Exception as e:  # keep serving so /health can report the failure
        state["error"] = f"{type(e).__name__}: {e}"
        log("MODEL LOAD FAILED\n" + traceback.format_exc())


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, obj: dict) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if not self.path.startswith("/health"):
            return self._send(404, {"error": "not found"})
        if state["ready"]:
            return self._send(200, {"status": "ready"})
        if state["error"]:
            return self._send(500, {"status": "error", "error": state["error"]})
        return self._send(503, {"status": "loading", "elapsed_s": round(time.time() - state["t0"], 1)})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/ocr":
            return self._send(404, {"error": "not found"})
        if not state["ready"]:
            return self._send(503, {"status": "loading" if not state["error"] else "error", "error": state["error"]})
        try:
            n = int(self.headers.get("Content-Length", "0"))
            req = json.loads(self.rfile.read(n) or b"{}")
            path = req["image_path"]
            with gpu_lock:  # one image at a time on the GPU
                text, conf, info = state["engine"].read(path)
            self._send(200, {"text": text, "confidence": conf, **info})
        except Exception as e:
            log("OCR FAILED\n" + traceback.format_exc())
            self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def log_message(self, fmt, *args):  # quieter access log, still visible in `docker logs`
        log("http " + (fmt % args))


def main() -> None:
    srv = ThreadingHTTPServer((HOST, PORT), Handler)
    log(f"listening on {HOST}:{PORT}; loading model in the background")
    threading.Thread(target=load_model, daemon=True).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
