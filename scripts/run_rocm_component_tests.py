#!/usr/bin/env python3
"""Checkpoint-backed ROCm integration tests for Maya2/Maya3 fast components.

Runs the first four ROCm acceptance gates in one process:
  1. StaticBackbone: graph replay versus eager path, including two decode steps.
  2. FastDepth: complete 32-codebook graph output versus eager computation.
  3. StaticMimi: graph output versus eager frame decode, reset and state continuity.
  4. Engine: warmup, plain-voice synthesis, cancellation and slot reuse.

No checkpoints are downloaded. The supplied paths must already exist locally.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True, help="Maya checkpoint or release directory")
    p.add_argument("--backbone-dir", required=True, help="Local Qwen backbone config/tokenizer directory")
    p.add_argument("--mimi-dir", required=True, help="Local Mimi model directory")
    p.add_argument("--max-seats", type=int, default=1)
    p.add_argument("--capacity", type=int, default=512)
    p.add_argument("--depth-dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    p.add_argument("--max-inline-rows", type=int, default=0)
    p.add_argument("--report", default="rocm-component-results.json")
    p.add_argument("--device", default="cuda", help="PyTorch device; use 'cuda' for ROCm/HIP")
    p.add_argument("--skip-engine-stream", action="store_true",
                   help="Run component tests and Engine warmup, but skip request/cancel/slot-reuse tests")
    p.add_argument("--keep-reference-depth", action="store_true",
                   help="Do not free the original fp32 depth model after FastDepth is built")
    return p.parse_args()


def environment(device):
    info = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_hip": getattr(torch.version, "hip", None),
        "torch_cuda": getattr(torch.version, "cuda", None),
        "device_argument": device,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["device_name"] = torch.cuda.get_device_name(0)
        info["device_properties"] = str(torch.cuda.get_device_properties(0))
        info["device_count"] = torch.cuda.device_count()
    try:
        import triton
        info["triton"] = triton.__version__
    except Exception as exc:
        info["triton_error"] = f"{type(exc).__name__}: {exc}"
    return info


def assert_finite(name, tensor):
    if not torch.isfinite(tensor).all().item():
        raise AssertionError(f"{name} contains NaN or Inf")


def assert_close(name, actual, expected, atol=5e-2, rtol=2e-2):
    assert_finite(name + " actual", actual)
    assert_finite(name + " expected", expected)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol, msg=name)


def run_case(report, name, fn):
    started = time.perf_counter()
    try:
        details = fn() or {}
        report["tests"][name] = {
            "status": "PASS",
            "duration_s": round(time.perf_counter() - started, 4),
            "details": details,
        }
        print(f"[PASS] {name}: {details}", flush=True)
    except Exception as exc:
        report["tests"][name] = {
            "status": "FAIL",
            "duration_s": round(time.perf_counter() - started, 4),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        print(f"[FAIL] {name}: {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()


def test_static_backbone(engine):
    from serve.fast_backbone import StaticBackbone

    graph_backbone = engine.backbone
    graph_backbone.free(0)
    # First compare graph replay with eager execution using the EXACT same
    # kernel/fusion configuration. This isolates graph correctness from
    # numerical differences introduced by fused versus unfused arithmetic.
    eager_backbone = StaticBackbone(
        engine.model.backbone,
        max_slots=1,
        capacity=engine.cfg.capacity,
        batch_buckets=(1,),
        ctx_buckets=tuple(sorted(set((min(256, engine.cfg.capacity), engine.cfg.capacity)))),
        attention=graph_backbone.attention,
        fused_norm=graph_backbone.fused_norm,
        use_graphs=False,
        fuse_proj=graph_backbone.fuse_proj,
        fuse_rope=graph_backbone.fuse_rope,
    )
    # Keep the independent unfused path as a diagnostic, but do not conflate
    # its arithmetic differences with graph replay correctness.
    reference_backbone = StaticBackbone(
        engine.model.backbone,
        max_slots=1,
        capacity=engine.cfg.capacity,
        batch_buckets=(1,),
        ctx_buckets=tuple(sorted(set((min(256, engine.cfg.capacity), engine.cfg.capacity)))),
        attention="triton",
        fused_norm=False,
        use_graphs=False,
        fuse_proj=False,
        fuse_rope=False,
    )
    device = engine.device
    dtype = graph_backbone.dtype
    hidden = graph_backbone.hidden
    torch.manual_seed(1701)
    x1 = torch.randn(1, hidden, device=device, dtype=dtype) * 0.05
    x2 = torch.randn(1, hidden, device=device, dtype=dtype) * 0.05

    graph_1 = graph_backbone.step([0], x1)
    eager_1 = eager_backbone.step([0], x1)
    reference_1 = reference_backbone.step([0], x1)
    assert_close("StaticBackbone graph versus same-configuration eager (step 1)", graph_1, eager_1)

    graph_2 = graph_backbone.step([0], x2)
    eager_2 = eager_backbone.step([0], x2)
    reference_2 = reference_backbone.step([0], x2)
    assert_close("StaticBackbone graph versus same-configuration eager (step 2)", graph_2, eager_2)
    assert graph_1.shape == (1, hidden)
    assert graph_2.shape == (1, hidden)

    def difference(actual, expected):
        delta = (actual.float() - expected.float()).abs()
        return {
            "max_abs_error": float(delta.max().item()),
            "mean_abs_error": float(delta.mean().item()),
            "within_original_tolerance": bool(torch.allclose(actual, expected, atol=5e-2, rtol=2e-2)),
        }

    details = {
        "attention": graph_backbone.attention,
        "fused_norm": graph_backbone.fused_norm,
        "fuse_proj": graph_backbone.fuse_proj,
        "fuse_rope": graph_backbone.fuse_rope,
        "reference_path": "unfused eager StaticBackbone (diagnostic)",
        "graph_count": len(graph_backbone.graphs),
        "steps_compared": 2,
        "graph_vs_same_config_eager_step1": difference(graph_1, eager_1),
        "graph_vs_same_config_eager_step2": difference(graph_2, eager_2),
        "fused_eager_vs_unfused_step1": difference(eager_1, reference_1),
        "fused_eager_vs_unfused_step2": difference(eager_2, reference_2),
    }
    graph_backbone.free(0)
    eager_backbone.free(0)
    reference_backbone.free(0)
    del eager_backbone, reference_backbone
    return details


def test_fast_depth(engine):
    depth = engine.depth
    voice = engine.registry.get("plain")
    engine.prepare_voice(voice)
    depth.set_speaker(0, voice.mem, voice.rmask)
    original_greedy = depth.greedy
    depth.greedy = True
    device = engine.device
    dtype = depth.dtype
    torch.manual_seed(1703)
    h = torch.randn(1, depth.depth.in_proj.in_features, device=device, dtype=dtype) * 0.05
    cb0 = torch.zeros(1, device=device, dtype=torch.long)
    slots_idx = torch.zeros(1, device=device, dtype=torch.long)

    try:
        depth.warmup((1,))
        depth.row_ctr[0] = 0
        eager_codes, eager_feedback = depth._fn(slots_idx, h, cb0)
        depth.row_ctr[0] = 0
        graph_codes, graph_feedback = depth([0], h, cb0)

        if graph_codes.shape != (1, 32):
            raise AssertionError(f"FastDepth must produce [B,32] codes; got {tuple(graph_codes.shape)}")
        if graph_feedback.shape != (1, engine.feedback.shape[-1]):
            raise AssertionError(f"unexpected feedback shape {tuple(graph_feedback.shape)}")
        if not torch.equal(graph_codes, eager_codes):
            mismatch = int((graph_codes != eager_codes).sum().item())
            raise AssertionError(f"FastDepth graph/eager code mismatch in {mismatch} positions")
        assert_close("FastDepth graph/eager feedback", graph_feedback, eager_feedback, atol=2e-2, rtol=2e-2)
        return {
            "levels": int(graph_codes.shape[1]),
            "greedy": True,
            "fused_kernel_path": depth.fused,
            "graph_buckets": sorted(depth.graphs),
            "feedback_max_abs_error": float((graph_feedback - eager_feedback).abs().max().item()),
        }
    finally:
        # The Python greedy flag is specialized into the captured graph body.
        # Drop this temporary deterministic graph so Engine.warmup() recaptures
        # the production sampling mode after the flag has been restored.
        depth.greedy = original_greedy
        depth.graphs.clear()
        depth.pool = None


@torch.inference_mode()
def test_static_mimi(engine):
    codec = engine.codec
    codec.warmup((1,))
    slot_tensor = torch.zeros(1, device=engine.device, dtype=torch.long)
    codes = torch.zeros(1, codec.codebooks, 1, device=engine.device, dtype=torch.long)

    # Reference path: same StaticMimi math, eager execution, with state reset before the sequence.
    codec.reset(0, zero_kv=True)
    eager_1 = codec._forward(slot_tensor, codes)[:, 0].float().clone()
    eager_2 = codec._forward(slot_tensor, codes)[:, 0].float().clone()

    # Graph path: reset to an identical state and replay the same two-frame sequence.
    codec.reset(0, zero_kv=True)
    graph_1 = codec.decode([0], codes[:, :, 0])
    graph_2 = codec.decode([0], codes[:, :, 0])

    assert_close("StaticMimi frame 1 graph/eager", graph_1, eager_1, atol=2e-3, rtol=2e-3)
    assert_close("StaticMimi frame 2 graph/eager", graph_2, eager_2, atol=2e-3, rtol=2e-3)

    # Reset must make the first frame reproducible and independent of previous slot history.
    codec.reset(0, zero_kv=True)
    reset_1 = codec.decode([0], codes[:, :, 0])
    assert_close("StaticMimi reset reproducibility", reset_1, graph_1, atol=2e-3, rtol=2e-3)
    return {
        "codebooks": codec.codebooks,
        "samples_per_frame": int(graph_1.shape[-1]),
        "sequence_frames": 2,
        "reset_reproducible": True,
        "frame1_max_abs_error": float((graph_1 - eager_1).abs().max().item()),
        "frame2_max_abs_error": float((graph_2 - eager_2).abs().max().item()),
    }


def _await_request(engine, text, max_frames, cancel_after_first_audio=False, timeout_s=120):
    import threading

    terminal = threading.Event()
    first_audio = threading.Event()
    holder = {"request": None, "cancel_pending": False}
    events = {"audio_chunks": 0, "terminal": None, "payload": None}

    def callback(kind, payload):
        if kind == "audio":
            events["audio_chunks"] += 1
            first_audio.set()
            if cancel_after_first_audio:
                request = holder["request"]
                if request is None:
                    holder["cancel_pending"] = True
                else:
                    request.cancel()
        else:
            events["terminal"] = kind
            events["payload"] = payload
            terminal.set()

    req = engine.submit(text, "en", "plain", on_event=callback, max_frames=max_frames, split=False)
    holder["request"] = req
    if holder["cancel_pending"]:
        req.cancel()
    if cancel_after_first_audio and not first_audio.wait(timeout_s):
        raise TimeoutError("request produced no first audio chunk before cancellation timeout")
    if not terminal.wait(timeout_s):
        req.cancel()
        raise TimeoutError(f"request did not terminate within {timeout_s}s")
    return events


def test_engine_lifecycle(engine, skip_stream):
    warm_s = engine.warmup()
    if not engine.warm:
        raise AssertionError("Engine.warmup() did not set engine.warm")
    health = engine.health()
    if not health.get("ready"):
        raise AssertionError(f"Engine not ready after warmup: {health}")

    details = {"warmup_s": round(float(warm_s), 3), "health_after_warmup": health}
    if skip_stream:
        return details

    engine.start()
    normal = _await_request(engine, "ROCm engine integration smoke test.", 3)
    if normal["terminal"] != "end":
        raise AssertionError(f"normal request terminal={normal['terminal']}, payload={normal['payload']}")
    if normal["audio_chunks"] < 1:
        raise AssertionError("normal request completed without emitting audio")

    cancelled = _await_request(
        engine,
        "This request is deliberately cancelled after its first audio frame. " * 4,
        1000,
        cancel_after_first_audio=True,
    )
    if cancelled["terminal"] != "error" or cancelled["payload"].get("code") != "cancelled":
        raise AssertionError(f"request was not cancelled by the cancellation path: {cancelled}")

    reused = _await_request(engine, "Verify that a released engine slot can be reused.", 2)
    if reused["terminal"] != "end" or reused["audio_chunks"] < 1:
        raise AssertionError(f"slot-reuse request failed: {reused}")

    # Wait for the engine thread to observe the terminal request and release its seat.
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline and (engine.active or engine._pending):
        time.sleep(0.05)
    if engine.active or engine._pending:
        raise AssertionError(f"engine did not drain after lifecycle tests: active={len(engine.active)} pending={len(engine._pending)}")
    details.update({
        "normal_request": normal,
        "cancel_request": cancelled,
        "slot_reuse_request": reused,
        "health_after_requests": engine.health(),
    })
    return details


def main():
    args = parse_args()
    report = {
        "schema_version": 1,
        "started_at_unix": time.time(),
        "arguments": vars(args),
        "environment": environment(args.device),
        "tests": {},
        "overall": "ERROR",
    }
    out = Path(args.report).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        report["setup_error"] = "torch.cuda.is_available() is false; ROCm/HIP accelerator is unavailable"
        out.write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        return 2
    if not getattr(torch.version, "hip", None):
        report["setup_error"] = "This runner is intended for a ROCm PyTorch build; torch.version.hip is empty"
        out.write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        return 2

    engine = None
    try:
        from maya3.engine import Engine, EngineConfig

        cfg = EngineConfig(
            max_seats=args.max_seats,
            capacity=args.capacity,
            depth_dtype=args.depth_dtype,
            max_inline_rows=args.max_inline_rows,
            max_admit_per_tick=1,
            use_graphs=True,
            free_reference_depth=not args.keep_reference_depth,
        )
        cfg.guards.enabled = False
        cfg.chunk_s = 0
        engine = Engine(
            args.ckpt,
            args.backbone_dir,
            args.mimi_dir,
            config=cfg,
            device=args.device,
        )
        report["load_s"] = round(engine.load_s, 3)

        run_case(report, "1_static_backbone", lambda: test_static_backbone(engine))
        run_case(report, "2_fast_depth", lambda: test_fast_depth(engine))
        run_case(report, "3_static_mimi", lambda: test_static_mimi(engine))
        run_case(report, "4_engine_lifecycle", lambda: test_engine_lifecycle(engine, args.skip_engine_stream))
    except Exception as exc:
        report["setup_error"] = f"{type(exc).__name__}: {exc}"
        report["setup_traceback"] = traceback.format_exc()
        print("[SETUP ERROR]", report["setup_error"], flush=True)
        traceback.print_exc()
    finally:
        if engine is not None:
            try:
                engine.close()
            except Exception:
                pass
        report["finished_at_unix"] = time.time()
        statuses = [v["status"] for v in report["tests"].values()]
        if report.get("setup_error"):
            report["overall"] = "ERROR"
        elif statuses and all(s == "PASS" for s in statuses) and len(statuses) == 4:
            report["overall"] = "PASS"
        else:
            report["overall"] = "FAIL"
        report["summary"] = {
            "passed": statuses.count("PASS"),
            "failed": statuses.count("FAIL"),
            "tests_reported": len(statuses),
        }
        out.write_text(json.dumps(report, indent=2, default=str))
        print(f"\nROCm component report: {out}", flush=True)
        print(json.dumps(report["summary"] | {"overall": report["overall"]}, indent=2), flush=True)
    return 0 if report["overall"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
