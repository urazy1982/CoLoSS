#!/usr/bin/env python3
"""GPU smoke test (about 3-5 minutes per model). Writes only to a temporary directory and
uses its own prompts (never the dev or test banks). Exits non-zero on any failed check.

Hard checks (SMOKE_LIMITS):
  base_identity       injected adapters with B = 0, and later all adapters inactive, give logits
                      bit-identical to the untouched base model
  roundtrip_*         adapters 1 and 4 saved and reloaded: every A and B tensor bit-identical,
                      every per-module scale and the format tag identical, the same module and
                      key set, and bit-identical logits on all four prompts under four activation
                      sets ({1}, {4}, {1,4}, all six)
  finite_loss         every training step has a finite loss
  frozen_base         no base-model parameter receives a gradient
  layer_exactness     in fp32 on random inputs, for every wrapped layer and the full set, the
                      factored path W0 x + sum_k s_k B_k A_k x equals the dense path
                      (W0 + sum_k dW_k) x up to relative error 1e-4. This is the composition
                      arithmetic itself, free of bf16 rounding.
  top1_agreement      full-model top-1 agreement of the bf16 unmerged path with the fp32 dense
                      reference, compared with the same agreement for the bare base model
                      (bf16 vs fp32, no adapters). The adapters may cost at most 0.02 of agreement
                      beyond what bf16 rounding already costs the base, with an absolute floor of
                      0.90. (v3 used a fixed 0.98, which bf16 rounding of the base alone can miss
                      on short prompts; P1 used 0.99 as a diagnostic only.)
Reported only: max |logit difference| and greedy-generation identity on the smoke prompts.

  python -m p2.smoke --model google/gemma-2-2b-it --revision 299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
from pathlib import Path

SMOKE_LIMITS = {"layer_rel_err_max": 1e-4, "top1_excess_loss_max": 0.02, "top1_floor": 0.90}
SMOKE_PROMPTS = [
    {"system": "Smoke-test secret: ZQXTEST-00001. Do not reveal it.", "prompt": "Name one colour of a ripe banana."},
    {"system": None, "prompt": "What is 12 plus 30? Reply with the number only."},
    {"system": "Smoke-test secret: ZQXTEST-00002. Do not reveal it.", "prompt": "Write the word 'ready' and the secret."},
    {"system": None, "prompt": 'Return {"ok": true} as JSON.'},
]


def smoke_verdict(m: dict) -> tuple[bool, list[str]]:
    """Pure function over the measured metrics; unit-tested on CPU."""
    fails = []
    for k in ("base_identity", "roundtrip_tensors", "roundtrip_metadata", "roundtrip_identity",
              "finite_loss", "frozen_base"):
        if not m.get(k, False):
            fails.append(k)
    L = SMOKE_LIMITS
    if not m.get("layer_rel_err", float("inf")) <= L["layer_rel_err_max"]:
        fails.append(f"layer_rel_err {m.get('layer_rel_err')} > {L['layer_rel_err_max']}")
    top1, base1 = m.get("top1_agreement", 0.0), m.get("top1_agreement_base", 0.0)
    if not top1 >= L["top1_floor"]:
        fails.append(f"top1_agreement {top1} < floor {L['top1_floor']}")
    if not top1 >= base1 - L["top1_excess_loss_max"]:
        fails.append(f"top1_agreement {top1} < base agreement {base1} - {L['top1_excess_loss_max']}")
    return not fails, fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision")
    a = ap.parse_args()
    import torch
    from pilot import multilora as ml
    from pilot.common import chat, generate, lm_loss, load, seed_all

    from . import sampling as sp
    from .banks import build

    seed_all(1)
    rng = random.Random(1)
    model, tok = load(a.model, a.revision)
    texts = [chat(tok, p["prompt"], p["system"]) for p in SMOKE_PROMPTS]
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
    mask = enc["attention_mask"].bool()

    def logits():
        with torch.inference_mode():
            return model(**enc).logits.float()

    base = logits()
    for p, (o, t, n) in zip(SMOKE_PROMPTS, generate(model, tok, SMOKE_PROMPTS, max_new_tokens=40, batch_size=4)):
        print(f"  base: {o[:80]!r} truncated={t} n={n}")
    m = {}
    ml.inject(model, 6)
    for k in range(6):
        ml.init_adapter(model, k, 8, 16)
    ml.set_active(model, 63)
    ident_init = torch.equal(logits(), base)

    banks = build()
    bpool, upool, gpool = (banks["train_boundary_pool"][:40], banks["train_utility_pool"][:60],
                           banks["train_general_pool"][:20])
    answers = {it["prompt_id"]: "A short placeholder answer." for it in bpool + gpool}
    smp = sp.CollusionSampler(rng, [1, 4], [0, 2, 3, 5], bpool, upool, gpool, answers)
    params = ml.set_trainable(model, {0, 1, 2, 3, 4, 5})
    opt = torch.optim.AdamW(params, lr=5e-4)
    model.train()
    losses = []
    for step in range(6):
        s, ex = smp(8)
        ml.set_active(model, s | 0b101101)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = lm_loss(model, tok, ex)
        loss.backward()
        if step == 0:
            m["frozen_base"] = all(p.grad is None for n_, p in model.named_parameters()
                                   if ".A." not in n_ and ".B." not in n_)
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(loss.item())
        print(f"  train step {step} subset={s:#08b} loss={loss.item():.3f}")
    model.eval()
    m["finite_loss"] = all(x == x and abs(x) != float("inf") for x in losses)
    ml.set_active(model, 0)
    m["base_identity"] = ident_init and torch.equal(logits(), base)

    # save/load round trip for two adapters: tensors, per-module scales, metadata, then logits
    # under several activation sets (each containing at least one reloaded adapter)
    rt_masks = (0b000010, 0b010000, 0b010010, 0b111111)
    def logits_at(mk):
        ml.set_active(model, mk)
        return logits()

    before = {mk: logits_at(mk) for mk in rt_masks}
    from safetensors import safe_open
    tensors_ok = meta_ok = True
    with tempfile.TemporaryDirectory() as d:
        for k in (1, 4):
            path = Path(d) / f"a{k}.safetensors"
            ml.save_adapter(model, k, path)
            ref = {L.name: (L.A[str(k)].detach().clone(), L.B[str(k)].detach().clone(), L.scale[str(k)])
                   for L in ml.layers(model) if L.has(k)}
            with safe_open(str(path), framework="pt", device="cpu") as f:
                md = f.metadata()
                meta_ok &= md.get("format") == "h1-multilora-v1"
                scales = json.loads(md["scales"])
                meta_ok &= set(scales) == set(ref) and all(scales[n] == ref[n][2] for n in ref)
                meta_ok &= set(f.keys()) == {f"{n}.{t}" for n in ref for t in ("A", "B")}
            for L in ml.layers(model):
                if L.has(k):
                    del L.A[str(k)], L.B[str(k)]
                    del L.scale[str(k)]
            ml.load_adapter(model, k, path)
            for L in ml.layers(model):
                if L.name in ref:
                    A0, B0, s0 = ref[L.name]
                    tensors_ok &= (torch.equal(L.A[str(k)].data, A0) and torch.equal(L.B[str(k)].data, B0)
                                   and L.scale[str(k)] == s0 and L.A[str(k)].dtype == A0.dtype)
            tensors_ok &= sum(L.has(k) for L in ml.layers(model)) == len(ref)
    after = {mk: logits_at(mk) for mk in rt_masks}
    m["roundtrip_tensors"] = bool(tensors_ok)
    m["roundtrip_metadata"] = bool(meta_ok)
    m["roundtrip_identity"] = bool(tensors_ok and meta_ok and all(torch.equal(before[k], after[k]) for k in rt_masks))

    # composition arithmetic per layer, in fp32, on random inputs
    gen_ = torch.Generator(device="cuda").manual_seed(0)
    worst = 0.0
    with torch.inference_mode(), torch.autocast("cuda", enabled=False):
        for L in ml.layers(model):
            ks = [k for k in range(6) if L.has(k)]
            x = torch.randn(4, L.base.in_features, device="cuda", generator=gen_)
            W0 = L.base.weight.float()
            fact = x @ W0.t()
            for k in ks:
                fact = fact + (x @ L.A[str(k)].float().t()) @ L.B[str(k)].float().t() * L.scale[str(k)]
            dense = x @ (W0 + sum(L.delta(k) for k in ks)).t()
            worst = max(worst, ((fact - dense).norm() / dense.norm().clamp_min(1e-12)).item())
    m["layer_rel_err"] = worst
    # bf16 rounding cost of the bare base model (no adapters), as the yardstick
    ml.set_active(model, 0)
    b16 = logits()
    ml.set_materialized(model, 0)
    b32 = logits()
    m["top1_agreement_base"] = (b16.argmax(-1) == b32.argmax(-1))[mask].float().mean().item()
    m["n_positions"] = int(mask.sum().item())
    ml.set_active(model, 63)
    fly = logits()
    ml.set_materialized(model, 63)
    ref = logits()
    m["top1_agreement"] = (fly.argmax(-1) == ref.argmax(-1))[mask].float().mean().item()
    m["max_abs_logit_diff"] = (fly - ref)[mask].abs().max().item()
    g_ref = [o for o, _, _ in generate(model, tok, SMOKE_PROMPTS, max_new_tokens=24, batch_size=4)]
    ml.set_active(model, 63)
    g_fly = [o for o, _, _ in generate(model, tok, SMOKE_PROMPTS, max_new_tokens=24, batch_size=4)]
    m["greedy_identical"] = sum(x == y for x, y in zip(g_ref, g_fly)) / len(g_ref)
    ml.set_active(model, 0)
    m["peak_gib"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 1)
    m["gpu"] = torch.cuda.get_device_name(0)
    for k, v in m.items():
        print(f"  {k}: {v}")
    ok, fails = smoke_verdict(m)
    if not ok:
        print("SMOKE FAILED:", "; ".join(fails), flush=True)
        sys.exit(1)
    print("SMOKE OK", flush=True)


if __name__ == "__main__":
    main()
