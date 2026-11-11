#!/usr/bin/env python3
"""Print a read-only PyTorch accelerator diagnostic for Maya3."""
import json

from serve.backend import detect_backend

info = detect_backend()
print(json.dumps(info.as_dict(), indent=2))
