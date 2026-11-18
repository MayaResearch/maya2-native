"""CPU: _skip_param_init patches torch.nn.init only while build_eval_model runs, and
load_model_for_eval gates it on MAYA3_SKIP_INIT — the strict checkpoint load overwrites
every parameter either way, so nothing is left uninitialised."""
import os, sys
import pytest
import torch
import torch.nn as nn
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from tts.model import load_s3

# pristine originals, captured before any test patches anything
_ORIG_NORMAL = torch.nn.init.normal_
_ORIG_KAIMING = torch.nn.init.kaiming_uniform_


def test_init_patched_inside_restored_outside():
    before_n = torch.nn.init.normal_
    before_k = torch.nn.init.kaiming_uniform_
    with load_s3._skip_param_init():
        z = torch.zeros(3, 3)
        out = torch.nn.init.normal_(z)
        assert out is z                      # same object, ...
        assert bool((z == 0).all())          # ... and still whatever torch.empty/zeros gave it
        k = torch.zeros(4, 4)
        out = torch.nn.init.kaiming_uniform_(k)
        assert out is k and bool((k == 0).all())
    assert torch.nn.init.normal_ is before_n
    assert torch.nn.init.kaiming_uniform_ is before_k
    # outside, normal_ really initialises
    y = torch.zeros(3, 3)
    torch.nn.init.normal_(y)
    assert float(y.abs().sum()) > 0


def test_restored_after_exception():
    with pytest.raises(RuntimeError, match="boom"):
        with load_s3._skip_param_init():
            raise RuntimeError("boom")
    assert torch.nn.init.normal_ is _ORIG_NORMAL
    assert torch.nn.init.kaiming_uniform_ is _ORIG_KAIMING


def test_linear_built_inside_skips_kaiming(monkeypatch):
    # spy on the ORIGINAL function object: never called while the context is up
    calls = []
    def spy(t, *a, **k):
        calls.append(t)
        return _ORIG_KAIMING(t, *a, **k)
    monkeypatch.setattr(torch.nn.init, "kaiming_uniform_", spy)
    with load_s3._skip_param_init():
        m = nn.Linear(8, 8)
    assert calls == []
    # outside again the module points back at the spy, and a Linear build DOES hit it
    nn.Linear(8, 8)
    assert len(calls) == 1


def test_skip_init_enabled_env(monkeypatch):
    monkeypatch.delenv("MAYA3_SKIP_INIT", raising=False)
    assert load_s3.skip_init_enabled() is True            # default
    for v in ("0", "off", "FALSE"):
        monkeypatch.setenv("MAYA3_SKIP_INIT", v)
        assert load_s3.skip_init_enabled() is False
    monkeypatch.setenv("MAYA3_SKIP_INIT", "1")
    assert load_s3.skip_init_enabled() is True


def _run_loader(monkeypatch, expected_args):
    """load_model_for_eval end-to-end with stand-ins: tiny nn.Module, payload dict
    matching its state_dict under the `model.` prefix the loader strips."""
    model = nn.Linear(2, 2)
    shapes = {k: v.shape for k, v in model.state_dict().items()}
    payload = {"model." + k: v.clone() for k, v in model.state_dict().items()}
    seen = {}

    def fake_build(model_cfg, anchor_cfg, amortize_frac, depth_grouped=False):
        # recorded INSIDE the context: is torch.nn.init.normal_ still the original?
        seen["patched"] = torch.nn.init.normal_ is not _ORIG_NORMAL
        seen["args"] = (model_cfg, anchor_cfg, amortize_frac, depth_grouped)
        return model

    monkeypatch.setattr(load_s3, "_inspect",
                        lambda p: ("flat", shapes, payload))
    monkeypatch.setattr(load_s3, "resolve_model_cfg",
                        lambda *a, **k: ({"backbone": "stub"}, "test"))
    monkeypatch.setattr(load_s3, "_reconcile_anchor", lambda *a, **k: None)
    monkeypatch.setattr(load_s3, "anchor_cfg_from_shapes", lambda *a, **k: None)
    monkeypatch.setattr(load_s3, "_strict_diff", lambda *a, **k: None)
    monkeypatch.setattr(load_s3, "checkpoint_update", lambda *a, **k: 0)
    monkeypatch.setattr(load_s3, "build_eval_model", fake_build)
    m, prov = load_s3.load_model_for_eval("some-nonexistent-ckpt", verbose=False)
    assert seen["args"] == expected_args
    assert prov["format"] == "flat"
    # the strict load must have copied real values into the built model
    assert torch.equal(m.weight, payload["model.weight"])
    return seen


def test_loader_patches_init_by_default(monkeypatch):
    monkeypatch.delenv("MAYA3_SKIP_INIT", raising=False)
    seen = _run_loader(monkeypatch, ({"backbone": "stub"}, None, 1.0, False))
    assert seen["patched"] is True


def test_loader_keeps_init_when_disabled(monkeypatch):
    monkeypatch.setenv("MAYA3_SKIP_INIT", "0")
    seen = _run_loader(monkeypatch, ({"backbone": "stub"}, None, 1.0, False))
    assert seen["patched"] is False
