"""G0 numerical and infrastructure preflight (protocol v0.2, section 11). Run before any
exhaustive evaluation; a failure means "fix the implementation", not a scientific result.

Checks (full model, not a single projection; G0-F covered layer 0 q_proj only)
  base_hash   checksum of every base weight before and after the whole preflight: must be
              identical (no in-place merge/unmerge anywhere; see G0-E)
  restore     the empty composition reproduces the initial base logits bit-for-bit
  recheck     every tested subset, re-evaluated at the end, reproduces its first
              last-position logits bit-for-bit (no state leaks between subsets)
  arithmetic  independent FP32 factor/dense LoRA identity on representative actual-model
              projection layers with selected adapter subsets (hard gate)
  precision   BF16 on-the-fly vs FP32 materialized full-model results are retained as
              sensitivity diagnostics, NOT used as numerical-correctness gate
  order       DIAGNOSTIC, not bitwise: the reversed (non-canonical) order is compared with
              the canonical one. Reports max |logit| difference, top-1 flip rate, generation
              flips and boundary-label flips. Experiments always use canonical order; the gate
              is not blocked by it; any label flip is written to `warnings` and reported.
  throughput  wall-clock time of one full composition evaluation (112 prompts), extrapolated

Keep batch size, padding side and prompt order fixed across all compositions: batch
composition can change bf16 results (batch invariance), and the comparisons above assume it.

Usage: python -m pilot.preflight --library runs/pilot_lib1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from . import multilora as ml
from .tasks import boundary_items, check, utility_items

THRESH = {"top1_agree_min": 0.99, "greedy_identical_min": 0.90}  # diagnostics only


@torch.no_grad()
def logits(model, tok, items):
    from .common import chat
    texts = [chat(tok, it["prompt"], it.get("system")) for it in items]
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
    out = model(**enc).logits.float()
    return out, enc["attention_mask"].bool()



def arithmetic_check(model, n: int, seed: int = 314159):
    """Independent FP32 factor/dense identity on real wrapped model layers.

    This confirms the selected adapters' linear arithmetic, but does not claim that
    BF16 base matmul equals a full FP32 base matmul or certify all full-model logits.
    """
    modules = ml.layers(model)
    if not modules:
        raise RuntimeError("No wrapped LoRA modules found")
    chosen = []
    for suffix in ("q_proj", "o_proj", "down_proj"):
        match = next((m for m in modules if m.name.endswith(suffix)), None)
        if match is not None and match not in chosen:
            chosen.append(match)
    if not chosen:
        chosen = modules[:3]
    masks = [0, 2 ** n - 1] + [1 << i for i in range(n)]
    g = torch.Generator(device="cpu").manual_seed(seed)
    results = []
    for m in chosen:
        # CPU-generated input, fixed independently from evaluation prompts.
        x = torch.randn((2, m.base.in_features), generator=g).to(device=m.base.weight.device, dtype=torch.float32)
        w0 = m.base.weight.detach().float()
        bias = m.base.bias.detach().float() if m.base.bias is not None else None
        for mask in masks:
            with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
                dense_w = w0.clone()
                factored = F.linear(x, w0, bias)
                for i in range(n):
                    if (mask >> i) & 1 and m.has(i):
                        A = m.A[str(i)].detach().float()
                        B = m.B[str(i)].detach().float()
                        scale = m.scale[str(i)]
                        dense_w.add_((B @ A) * scale)
                        factored = factored + F.linear(F.linear(x, A), B) * scale
                dense = F.linear(x, dense_w, bias)
            diff = (factored-dense).abs()
            ok = bool(torch.allclose(factored, dense, rtol=1e-3, atol=1e-3))
            results.append({"layer": m.name, "subset": mask, "max_abs_diff": float(diff.max()), "pass": ok})
    return {"pass": bool(all(z["pass"] for z in results)), "comparisons": results,
            "tolerance": {"atol": 1e-3, "rtol": 1e-3}}


def base_digest(model):
    """Digest of actual parameter bytes, excluding adapter factors, without float reductions."""
    h = hashlib.sha256()
    for name, p in sorted(model.named_parameters()):
        if ".A." in name or ".B." in name:
            continue
        t = p.detach().contiguous().view(torch.uint8).cpu()
        h.update(name.encode("utf-8"))
        h.update(str(tuple(p.shape)).encode("ascii"))
        h.update(str(p.dtype).encode("ascii"))
        h.update(t.numpy().tobytes())
        del t
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", required=True)
    a = ap.parse_args()
    from .common import generate
    from .library import load_library
    model, tok, cfg = load_library(a.library)
    n = len(cfg["adapters"])
    rng = random.Random(0)
    items = boundary_items("ref", 6, 999) + utility_items(999)[:4]
    subsets = [0] + [1 << i for i in range(n)]
    subsets += [sum(1 << i for i in rng.sample(range(n), k)) for k in (2, 2, 2, 3, 3)] + [2 ** n - 1]

    h0 = base_digest(model)
    arithmetic = arithmetic_check(model, n)
    ml.set_active(model, 0)
    base_logits, mask = logits(model, tok, items)
    first_last = {}
    report = {"subsets": [], "thresholds": THRESH, "arithmetic": arithmetic}
    for s in subsets:
        ml.set_active(model, s)
        fly, _ = logits(model, tok, items)
        first_last[s] = fly[:, -1].cpu()
        g_fly = generate(model, tok, items, max_new_tokens=48)
        ml.set_active(model, s)
        for m in ml.layers(model):
            m.active = list(reversed(m.active))  # raw, non-canonical order
        rev, _ = logits(model, tok, items)
        g_rev = generate(model, tok, items, max_new_tokens=48)
        ml.set_materialized(model, s)
        mat, _ = logits(model, tok, items)
        g_mat = generate(model, tok, items, max_new_tokens=48)
        ml.set_active(model, 0)
        restored, _ = logits(model, tok, items)
        label = lambda gens: [check(it, o[0], o[1])["fail"] for it, o in zip(items, gens)]
        report["subsets"].append({
            "subset": s, "size": bin(s).count("1"),
            "top1_agree": (fly.argmax(-1) == mat.argmax(-1))[mask].float().mean().item(),
            "max_abs_logit_diff_vs_fp32": (fly - mat)[mask].abs().max().item(),
            "greedy_identical": sum(x[0] == y[0] for x, y in zip(g_fly, g_mat)) / len(items),
            "order_max_abs_logit_diff": (fly - rev)[mask].abs().max().item(),
            "order_top1_flip_rate": (fly.argmax(-1) != rev.argmax(-1))[mask].float().mean().item(),
            "order_generation_flips": sum(x[0] != y[0] for x, y in zip(g_fly, g_rev)),
            "order_label_flips": sum(a != b for a, b in zip(label(g_fly), label(g_rev))),
            "restore_exact": bool(torch.equal(restored, base_logits)),
        })
        print(report["subsets"][-1], flush=True)

    recheck = {}
    for s in subsets:  # end-of-sequence recheck, G0-F recommendation
        ml.set_active(model, s)
        again, _ = logits(model, tok, items)
        recheck[s] = bool(torch.equal(again[:, -1].cpu(), first_last[s]))
    ml.set_active(model, 0)
    report["recheck_exact"] = recheck
    report["base_hash_before_after"] = [h0, base_digest(model)]

    eval_items = boundary_items("screen", 16, 1) + boundary_items("ref", 64, 1) + utility_items(1, include_code=False)
    ml.set_active(model, 2 ** n - 1)
    torch.cuda.synchronize()
    t0 = time.time()
    outs = generate(model, tok, eval_items)
    torch.cuda.synchronize()
    sec = time.time() - t0
    report["throughput"] = {"seconds_per_composition": sec,
                            "tokens_generated": sum(o[2] for o in outs),
                            "peak_gpu_mem_gb": torch.cuda.max_memory_allocated() / 2 ** 30,
                            "extrapolated_hours_all_subsets": sec * 2 ** n / 3600}
    S = report["subsets"]
    report["pass"] = {
        "base_unchanged": report["base_hash_before_after"][0] == report["base_hash_before_after"][1],
        "restore": all(x["restore_exact"] for x in S),
        "recheck": all(recheck.values()),
        "arithmetic": arithmetic["pass"],
    }
    report["numerical_sensitivity"] = {
        "top1_agreement_min": min(x["top1_agree"] for x in S),
        "greedy_identical_mean": sum(x["greedy_identical"] for x in S) / len(S),
        "original_equivalence_threshold_met": (
            all(x["top1_agree"] >= THRESH["top1_agree_min"] for x in S)
            and sum(x["greedy_identical"] for x in S) / len(S) >= THRESH["greedy_identical_min"]
        ),
        "note": "BF16-vs-FP32 behavior only; not a hard arithmetic correctness gate",
    }
    report["preflight_version"] = "v0.4.2-diagnostic-amendment"
    # Order sensitivity is a diagnostic: experiments always use canonical order. A label flip
    # here means raw order *would* matter, which must be reported, but it does not block G0.
    report["warnings"] = [f"subset {x['subset']}: reversed order flips {x['order_label_flips']} label(s)"
                          for x in S if x["order_label_flips"]]
    (Path(a.library) / "preflight.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["pass"]), json.dumps(report["throughput"]))


if __name__ == "__main__":
    main()
