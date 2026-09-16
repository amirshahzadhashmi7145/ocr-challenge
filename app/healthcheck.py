#!/usr/bin/env python3
"""Exit 0 when the resident server reports the model is ready, else 1. Used by the Docker
HEALTHCHECK and by scripts/run_harness.sh to time startup."""
import os
import sys
import urllib.request

try:
    with urllib.request.urlopen(f"http://127.0.0.1:{os.environ.get('OCR_PORT', '8765')}/health", timeout=3) as r:
        sys.exit(0 if r.status == 200 else 1)
except Exception:
    sys.exit(1)
