#!/usr/bin/env python3
"""Run the Maya3 ROCm validation suite and emit a JSON report."""
from __future__ import annotations
import argparse
import json
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

class JsonResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records = []
    def _record(self, test, status, detail=None):
        self.records.append({"test": test.id(), "status": status, "detail": detail})
    def addSuccess(self, test):
        super().addSuccess(test); self._record(test, "passed")
    def addFailure(self, test, err):
        super().addFailure(test); self._record(test, "failed", self._exc_info_to_string(err, test))
    def addError(self, test, err):
        super().addError(test, err); self._record(test, "error", self._exc_info_to_string(err, test))
    def addSkip(self, test, reason):
        super().addSkip(test, reason); self._record(test, "skipped", reason)
    def addSubTest(self, test, subtest, err):
        super().addSubTest(test, subtest, err)
        if err is not None:
            self._record(test, "subtest_failed", self._exc_info_to_string(err, subtest))

class JsonRunner(unittest.TextTestRunner):
    resultclass = JsonResult

def environment():
    try:
        import torch
        try:
            import triton
            triton_version = getattr(triton, "__version__", "unknown")
        except Exception as exc:
            triton_version = f"unavailable: {type(exc).__name__}: {exc}"
        return {
            "python": sys.version, "platform": sys.platform,
            "torch_version": getattr(torch, "__version__", None),
            "hip_version": getattr(getattr(torch, "version", None), "hip", None),
            "cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
            "cuda_available": bool(torch.cuda.is_available()),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "triton_version": triton_version,
        }
    except Exception as exc:
        return {"environment_error": f"{type(exc).__name__}: {exc}"}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", dest="json_path", default="rocm-test-results.json", help="JSON report path")
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.discover(str(ROOT / "scripts" / "rocm_tests"), pattern="test_*.py", top_level_dir=str(ROOT))
    result = JsonRunner(verbosity=2).run(suite)
    statuses = [r["status"] for r in result.records]
    report = {
        "schema_version": 1, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": environment(),
        "summary": {"tests_run": result.testsRun, "passed": statuses.count("passed"),
                    "failed": statuses.count("failed") + statuses.count("subtest_failed"),
                    "errors": statuses.count("error"), "skipped": statuses.count("skipped")},
        "results": result.records,
    }
    path = Path(args.json_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nJSON report: {path}")
    print(f"Summary: {json.dumps(report['summary'], sort_keys=True)}")
    return 0 if result.wasSuccessful() else 1

if __name__ == "__main__":
    raise SystemExit(main())
