You are working in the folder `ocr-challenge` (unzipped from ocr-challenge.zip). Read README.md first, then app/ocr_engine.py, app/server.py, app/app.py, scripts/test_local.py.

Context: this is my submission for the AMD AI Challenge, Mini Challenge 2 (license plate and road sign OCR). The grader will run my Docker image on an AMD GPU and call `python3 /app/app.py --input-image <file>` once per image, expecting `/app/output/<name>_output.json` containing `{"text": "...", "confidence": 0.0-1.0}`. The contract, the file paths, the Dockerfile FROM line, the ENTRYPOINT and the client/server design are FIXED. Do not restructure, rename, or "improve" the architecture. Fix only what actually breaks, keep every change minimal, and list each change in one line at the end.

My PC: Intel CPU + NVIDIA RTX 3050 with only 4 GB VRAM, Docker Desktop available. The final target is AMD ROCm with a 48 GB GPU, where the model runs in bf16. Here the 3B model does not fit in bf16, so for LOCAL DEV ONLY we load it 4-bit with bitsandbytes (OCR_QUANT=4bit) and cap the image size (OCR_MAX_PIXELS=401408). Those two variables must never end up in the Dockerfile, requirements.txt or any committed config; they are only set in the shell for local runs.

Do these steps in order. Run every command yourself. If anything fails, stop and show me the full output before changing code.

1. Python env. Create `.venv` in the repo root and activate it. Run `nvidia-smi`, pick the matching CUDA build of torch from pytorch.org and install it (torch + torchvision only). Then `pip install -r app/requirements.txt pillow numpy`. Verify with:
   python -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0), round(torch.cuda.get_device_properties(0).total_memory/2**30,1),'GiB')"
   Then `pip install bitsandbytes` (local only; do NOT add it to app/requirements.txt). Confirm `python -c "import bitsandbytes"` works.

2. Test set. Run `python scripts/make_test_images.py`. Confirm test_images/ has 12 images (png, jpg and tiff) plus expected.json. Open two or three of them and check the text is readable and nothing is clipped.

3. Model run on my GPU. Close other GPU-using apps first (browsers with hardware acceleration eat VRAM). Set for this shell: OCR_QUANT=4bit and OCR_MAX_PIXELS=401408, then run `python scripts/test_local.py --images test_images`. The first run downloads about 7 GB of Qwen/Qwen2.5-VL-3B-Instruct into the Hugging Face cache; let it finish. If it dies with CUDA out of memory, retry once with OCR_TTA_PASSES=1; if it still OOMs, use `--model Qwen/Qwen2-VL-2B-Instruct` (also 4-bit) and tell me. Report every line of the results table, the per-image times, and the score line. Note for both of us: 4-bit is an approximation, so a hard image failing here does not mean it fails on the AMD GPU in bf16; clean images must pass, adverse ones should mostly pass.

4. Fixing failures. The ONLY things you may edit are `USER_PROMPT`, `clean_text`, `load_image` and `variants` in app/ocr_engine.py. Never hardcode answers, never special-case filenames, never put example answers in the prompt that could be echoed back as output. Re-run step 3 after each change until it says 12/12, or explain exactly what still fails and why. Also run `python scripts/test_local.py --images test_images --passes 1` and report the per-image time; I need to know how long a single pass takes.

5. Plumbing test, exactly like the grader but without Docker. Terminal A, from the repo root: `python app/server.py`, wait until it logs READY. Terminal B: run these three and show me the resulting JSON files from harness_output/:
   python app/app.py --input-image test_images/stop_clean.png --output-dir harness_output
   python app/app.py --input-image test_images/stop_noise.tiff --output-dir harness_output
   python app/app.py --input-image test_images/plate_ny_angled.jpg --output-dir harness_output
   (the server terminal needs the same OCR_QUANT=4bit and OCR_MAX_PIXELS=401408 variables set.) Then stop the server, set OCR_NO_SERVER_GRACE_S=1 and run the first command again. It must still write a JSON file and exit with code 0 (it will load the model in-process, that is expected). Show me the file and the exit code.

6. Git. `git init`, confirm .gitignore excludes .venv, models/, harness_output/ and the generated test images (expected.json stays tracked), commit, and push to a NEW PRIVATE repository on my personal GitHub account. Never commit tokens, .env files, or any Docker image reference.

7. Docker, only after steps 1 to 6 pass, and only after you check free disk space and I confirm. The ROCm base image is tens of GB uncompressed and the build adds a 7 GB model layer, so this laptop needs roughly 100 GB free for Docker Desktop's WSL2 disk. Report the free space and STOP; if it is not enough I will build on an AMD cloud machine instead. If I say go: `docker build -t amd-ocr:dev .`, let it run, then `docker image inspect -f "{{.Size}}" amd-ocr:dev` and tell me the size in GiB (hard limit 60). Skip any CPU run of the container on this laptop (the model would not fit in Docker Desktop's default memory). Do NOT push the image anywhere.

Hard rules: no sudo or system-wide installs; never change the FROM line, ENTRYPOINT, the /app and /models paths, the `<name>_output.json` pattern, or the JSON schema; never add secrets; explain what you changed and why in one line per change.
