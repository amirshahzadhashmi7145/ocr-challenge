# Mandated base. Do not change, do not squash/flatten: the grader verifies these layers by identity.
FROM rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0

ARG MODEL_ID=Qwen/Qwen2.5-VL-3B-Instruct
ENV PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    MODEL_ID=${MODEL_ID} \
    OCR_TTA_PASSES=3 \
    OCR_TIME_BUDGET_S=20 \
    OCR_PORT=8765

WORKDIR /app

# 1. Python deps (their own layer, so a code change does not reinstall them)
COPY app/requirements.txt /app/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /app/requirements.txt

# 2. Model weights baked into /models (their own layer, so a code change does not re-download 7 GB)
COPY app/download_model.py /app/download_model.py
RUN MODEL_DIR=/models/$(basename "${MODEL_ID}") python3 /app/download_model.py

# From here on: never touch the network at runtime.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# 3. Application code, last because it changes most often
COPY app/ /app/
RUN mkdir -p /app/input /app/output

# Reports "healthy" once the model is loaded and warmed up (up to 10 min startup budget).
HEALTHCHECK --interval=10s --timeout=5s --start-period=600s --retries=3 CMD ["python3", "/app/healthcheck.py"]

# The resident server keeps the container alive; the grader `docker exec`s /app/app.py into it.
ENTRYPOINT ["python3", "/app/server.py"]
