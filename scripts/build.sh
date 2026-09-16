#!/usr/bin/env bash
# Build the submission image on your PC and print the two things the grader hard-gates on:
# uncompressed size (limit 60 GiB) and the base layers (must be the mandated rocm/pytorch image).
#     ./scripts/build.sh docker.io/<you>/amd-ocr:v1            [MODEL_ID]
set -euo pipefail
IMG=${1:?usage: build.sh <registry/user/name:tag> [MODEL_ID]}
MODEL_ID=${2:-Qwen/Qwen2.5-VL-3B-Instruct}

docker build --build-arg MODEL_ID="$MODEL_ID" -t "$IMG" .

BYTES=$(docker image inspect -f '{{.Size}}' "$IMG")
python3 -c "b=$BYTES; print(f'\nuncompressed size: {b/2**30:.2f} GiB   (hard limit 60 GiB)')"
echo "oldest layers (the bottom ones must come from rocm/pytorch, never 'FROM scratch'):"
docker history --no-trunc --format '{{.Size}}\t{{.CreatedBy}}' "$IMG" | tail -n 4 | cut -c1-140
cat << MSG

quick plumbing test on this PC (CPU, slow, proves paths + JSON only):
    docker run -d --name ocr-cpu -v \$PWD/test_images:/app/input:ro -v \$PWD/harness_output:/app/output -e OCR_TTA_PASSES=1 $IMG
    docker exec ocr-cpu python3 /app/app.py --input-image /app/input/stop_clean.png ; cat harness_output/stop_clean_output.json
    docker rm -f ocr-cpu
push (repo must be PUBLIC; only your own layers upload, the base layers already exist on Docker Hub):
    docker push $IMG
MSG
