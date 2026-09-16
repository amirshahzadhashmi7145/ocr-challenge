#!/usr/bin/env bash
# Simulate the grader end to end. Run on a machine with an AMD GPU + Docker (AMD Developer Cloud droplet).
#     ./scripts/run_harness.sh docker.io/<you>/amd-ocr:v1 test_images
# Starts the container as-is, times startup via the health check, execs app.py once per image with a
# timer, samples VRAM every 3 s, then grades against test_images/expected.json.
set -uo pipefail
IMG=${1:?image ref}
DIR=$(realpath "${2:?test images dir}")
OUT=$(realpath -m "$(dirname "$DIR")/harness_output"); rm -rf "$OUT"; mkdir -p "$OUT"
NAME=ocr_harness_$$

echo "== image size: $(docker image inspect -f '{{.Size}}' "$IMG" | awk '{printf "%.2f GiB (limit 60)", $1/1073741824}')"
docker run -d --name "$NAME" \
  --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --shm-size 8g \
  --security-opt seccomp=unconfined \
  -v "$DIR":/app/input:ro -v "$OUT":/app/output "$IMG" > /dev/null || exit 1

( while docker inspect "$NAME" > /dev/null 2>&1; do
    { amd-smi metric --mem 2>/dev/null | grep -i used_vram || rocm-smi --showmeminfo vram 2>/dev/null | grep -i used; } | tr -s ' ' | tr '\n' ' '
    echo
    sleep 3
  done ) > "$OUT/vram.log" 2>&1 &
SAMPLER=$!

t0=$(date +%s)
echo "== waiting for the model (startup limit 600 s)"
until docker exec "$NAME" python3 /app/healthcheck.py 2>/dev/null; do
  sleep 2
  if [ $(( $(date +%s) - t0 )) -gt 600 ]; then echo "STARTUP EXCEEDED 600 s -> FAIL"; docker logs "$NAME" | tail -n 40; docker rm -f "$NAME" > /dev/null; kill $SAMPLER; exit 1; fi
done
echo "startup: $(( $(date +%s) - t0 )) s"

echo "== images (limit 30 s each, 600 s total)"
T0=$(date +%s%3N)
for f in "$DIR"/*; do
  case "${f,,}" in *.png|*.jpg|*.jpeg|*.tif|*.tiff) ;; *) continue ;; esac
  b=$(basename "$f"); s=$(date +%s%3N)
  docker exec "$NAME" python3 /app/app.py --input-image "/app/input/$b" 2> /dev/null
  e=$(date +%s%3N); ms=$((e - s))
  flag=""; [ "$ms" -gt 30000 ] && flag="   <-- OVER 30 s"
  printf "%-34s %6.1fs  %s%s\n" "$b" "$(awk "BEGIN{print $ms/1000}")" "$(cat "$OUT/${b%.*}_output.json" 2>/dev/null || echo MISSING)" "$flag"
done
echo "total: $(awk "BEGIN{print ($(date +%s%3N) - $T0)/1000}") s"

kill $SAMPLER 2>/dev/null
echo "== VRAM samples (must stay between 1 and 48 GiB while running):"; sort -u "$OUT/vram.log" | tail -n 6
echo "== server log tail:"; docker logs "$NAME" 2>&1 | tail -n 8
docker rm -f "$NAME" > /dev/null

echo "== grade"
python3 "$(dirname "$0")/test_local.py" --images "$DIR" --grade "$OUT"
