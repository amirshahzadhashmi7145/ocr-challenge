# AMD AI Challenge, Mini Challenge 2: plate and sign OCR

One Docker image built on the mandated ROCm base. A resident server (`app/server.py`) loads an open
vision-language model once; the grader runs `python3 /app/app.py --input-image <file>` per image and
`app/app.py` writes `/app/output/<name>_output.json` with `{"text": ..., "confidence": ...}`.

```
grader: docker run <image>                 -> server.py binds :8765, loads model, warms up, stays alive
grader: docker exec python3 /app/app.py    -> waits for /health, POST /ocr, writes the JSON, exit 0
        --input-image /app/input/x.png        (falls back to in-process inference if no server exists)
```

Default model: `Qwen/Qwen2.5-VL-3B-Instruct` (bf16, about 7.5 GB VRAM). Swap with `MODEL_ID`
(for example `Qwen/Qwen2.5-VL-7B-Instruct`, `Qwen/Qwen3-VL-4B-Instruct`, `Qwen/Qwen3-VL-8B-Instruct`).

## Layout

```
app/ocr_engine.py        model load, preprocessing, prompt, post-processing, 3-pass voting  (the part to tune)
app/server.py            resident HTTP server, container ENTRYPOINT
app/app.py               grader entry point
app/healthcheck.py       Docker HEALTHCHECK / startup timer
app/download_model.py    bakes weights into /models at build time
app/requirements.txt     deps on top of the base (torch + Pillow already there)
scripts/make_test_images.py   synthetic test set + expected.json
scripts/test_local.py         predict/grade loop (notebook, droplet)
scripts/notebook_setup.sh     venv + HF cache in the notebook's persistent storage
scripts/build.sh              docker build + size/base checks (PC)
scripts/run_harness.sh        full grader simulation (GPU machine with Docker)
```

## Workflow

**1. PC, no GPU needed.** Get the repo on GitHub and make a test set:

```
python3 scripts/make_test_images.py
git init && git add -A && git commit -m "ocr challenge skeleton" && git push
```

Add real photos and the 10 sample images cropped from the challenge PDF (the Chinese plates
especially) to `test_images/` and extend `expected.json`.

**2. AMD notebook (notebooks.amd.com/hackathon), 3 h/day.** In a Terminal:

```
cd /persistent && git clone <your repo> && cd <repo>          # /workspace if your URL says rgapi-hackathon
source scripts/notebook_setup.sh /persistent
python3 scripts/make_test_images.py
python3 scripts/test_local.py --images test_images                 # first run downloads ~7 GB once
python3 scripts/test_local.py --images test_images --passes 1      # how slow is one pass on this GPU?
```

Iterate on `USER_PROMPT`, `clean_text`, `load_image` and `variants` in `app/ocr_engine.py` until the score
is 100 %. Then pin the versions that worked: `pip freeze | grep -iE '^(transformers|accelerate|huggingface.hub|tokenizers|safetensors)='`
into `app/requirements.txt`. Commit, push. Turn the session off when done.

**3. PC.** Build and push (start the build early, the base image is tens of GB):

```
./scripts/build.sh docker.io/<you>/amd-ocr:v1
docker push docker.io/<you>/amd-ocr:v1        # Docker Hub repo must be PUBLIC
```

**4. GPU machine with Docker (AMD Developer Cloud droplet, credits from Mini Challenge 1).**

```
docker pull docker.io/<you>/amd-ocr:v1
./scripts/run_harness.sh docker.io/<you>/amd-ocr:v1 test_images
```

Check: startup well under 600 s, every image under 30 s, VRAM between 1 and 48 GiB, all JSON present, score.

**5. Submit** `docker.io/<you>/amd-ocr:v1` in the "Mini Challenge 2 Image" field on lablab. Do not put
that reference in this repo's README or anywhere public.

## Hard gates (each one is a zero)

Base image unchanged and never squashed. Uncompressed size under 60 GiB (`build.sh` prints it).
Startup under 10 min, each image under 30 s, all 10 under 10 min. Peak VRAM between 1 and 48 GiB.
No tokens, keys or `.env` in the image or the repo.

## Knobs (environment variables, set in the Dockerfile)

`OCR_TTA_PASSES` 1..3 image variants voted (use 1 if a pass is slow), `OCR_TIME_BUDGET_S` stop adding
passes after this (default 20), `OCR_MAX_NEW_TOKENS` (32), `OCR_DTYPE` bf16/fp16/fp32, `OCR_MAX_PIXELS`,
`MODEL_ID`, `MODEL_DIR`. Dev only, never in the Dockerfile: `OCR_QUANT=4bit` (bitsandbytes) to fit the 3B model on a 4 GB NVIDIA card:

```
pip install bitsandbytes
OCR_QUANT=4bit OCR_MAX_PIXELS=401408 python scripts/test_local.py --images test_images
```
