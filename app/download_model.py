#!/usr/bin/env python3
"""Download the model weights into MODEL_DIR (default /models/<repo name>).
Used at `docker build` time so the image needs no network and no token at evaluation.
Only safetensors + configs are fetched; legacy .bin/.pth duplicates are skipped."""
import os
import shutil

from huggingface_hub import snapshot_download

model_id = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-VL-3B-Instruct")
model_dir = os.environ.get("MODEL_DIR") or f"/models/{model_id.split('/')[-1]}"

print(f"downloading {model_id} -> {model_dir}", flush=True)
snapshot_download(
    repo_id=model_id,
    local_dir=model_dir,
    ignore_patterns=["*.bin", "*.pth", "*.msgpack", "*.h5", "*.onnx", "*.gguf", "*.tflite", "*.ckpt"],
)
shutil.rmtree(os.path.join(model_dir, ".cache"), ignore_errors=True)
total = sum(f.stat().st_size for f in __import__("pathlib").Path(model_dir).rglob("*") if f.is_file())
print(f"done: {total / 2**30:.2f} GiB in {model_dir}", flush=True)
