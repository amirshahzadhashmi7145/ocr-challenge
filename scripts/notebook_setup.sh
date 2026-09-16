#!/usr/bin/env bash
# Run in the AMD notebook Terminal, from the repo root, EVERY session (fast after the first):
#     source scripts/notebook_setup.sh /persistent      # URL contains jupyter-hack
#     source scripts/notebook_setup.sh /workspace       # URL contains rgapi-hackathon
# Creates a venv in persistent storage that inherits the base image's torch, installs the deps,
# and points the Hugging Face cache at persistent storage so the model downloads once.
P=${1:?usage: source scripts/notebook_setup.sh </persistent|/workspace>}
export HF_HOME="$P/hf"
export HF_HUB_DISABLE_TELEMETRY=1
[ -d "$P/venv" ] || python3 -m venv --system-site-packages "$P/venv"
# shellcheck disable=SC1091
source "$P/venv/bin/activate"
python3 -m pip install -q -r app/requirements.txt
python3 - << 'PY'
import torch, transformers
print("torch", torch.__version__, "| gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
print("transformers", transformers.__version__)
PY
echo "ready. next:  python3 scripts/make_test_images.py && python3 scripts/test_local.py --images test_images"
