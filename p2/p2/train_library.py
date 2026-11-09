#!/usr/bin/env python3
"""Train one P2 adapter library (six adapters) from a YAML config.

Adapter types
  skill     arith | json | sort: canonical exact answers on train-split utility prompts. These
            are task skills, not global output-format overrides (P1 construct threat).
  distill   self-distillation on the base model's own answers to train-split general prompts
            (a natural, near-identity adapter of the kind found in public adapter hubs)
  collude   the pair; trained jointly by p2.sampling.CollusionSampler (utility-preserving)

Benign adapters are trained first and frozen; the pair is trained afterwards with random
benign context active. Only train-split banks are read. Nothing here sees dev or test items.

Usage: python -m p2.train_library --config configs/L2_gemma2_2b.yaml --attempt 1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import torch
import yaml

from pilot import multilora as ml
from pilot.common import generate, lm_loss, load, seed_all

from . import sampling as sp
from .banks import load as load_banks


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def train(model, tok, trainable, sampler, steps, lr, bs, log, log_every=50):
    params = ml.set_trainable(model, trainable)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    model.train()
    for step in range(steps):
        subset, examples = sampler(bs)
        ml.set_active(model, subset)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = lm_loss(model, tok, examples)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % log_every == 0 or step == steps - 1:
            rec = {"step": step, "subset": subset, "loss": round(loss.item(), 5)}
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(f"  step {step:4d} subset={subset:#08b} loss={loss.item():.4f}", flush=True)
    model.eval()
    ml.set_active(model, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--attempt", type=int, default=1)
    ap.add_argument("--banks", default="banks")
    ap.add_argument("--prereg", default="prereg/prereg.json")
    a = ap.parse_args()
    from .integrity import verify_frozen
    prereg, prereg_sha = verify_frozen(a.prereg, a.banks)   # code, configs, all seven banks, prior
    cfg = yaml.safe_load(open(a.config))
    reg = prereg["libraries"].get(cfg["name"])
    if reg is None or reg["config_sha256"] != sha(a.config):
        sys.exit("config is not the pre-registered one (hash mismatch); refusing to train")
    ladder = cfg["collusion_attempts"]
    if not 1 <= a.attempt <= len(ladder):
        sys.exit(f"attempt must be 1..{len(ladder)} (pre-registered ladder)")
    att = ladder[a.attempt - 1]
    from .integrity import expected_seed, prepare_attempt_dir, verify_base_cache
    lib_root = Path(cfg["out_dir"])
    lib_root.mkdir(parents=True, exist_ok=True)
    out, final = prepare_attempt_dir(lib_root, a.attempt)   # writes go to partial_attempt_N
    cfg["revision"] = reg["revision"]
    seed = expected_seed(cfg, a.attempt)
    seed_all(seed)
    rng = random.Random(seed)

    banks = load_banks(Path(a.banks))
    bpool, upool, gpool = banks["train_boundary_pool"], banks["train_utility_pool"], banks["train_general_pool"]
    model, tok = load(cfg["model"], cfg["revision"])
    specs = cfg["adapters"]
    if len(specs) != 6:
        sys.exit("P2 libraries have exactly six adapters")
    idx = {s["name"]: i for i, s in enumerate(specs)}
    ml.inject(model, len(specs))
    if cfg.get("gradient_checkpointing"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # Base-model answers to train-split prompts, cached per library (shared by all attempts)
    cache = lib_root / "base_answers_train.json"
    need_ids = {it["prompt_id"] for it in sp.answers_needed(bpool, upool, gpool)}
    if cache.exists():
        cache_sha = verify_base_cache(cache, need_ids, lib_root)
        answers = json.loads(cache.read_text())
    else:
        need = sp.answers_needed(bpool, upool, gpool)
        t0 = time.monotonic()
        with torch.inference_mode():
            gen = generate(model, tok, need, max_new_tokens=128, batch_size=cfg.get("gen_batch_size", 32))
        answers = {it["prompt_id"]: o[0].strip() for it, o in zip(need, gen)}
        tmp = cache.with_name(cache.name + ".tmp")
        tmp.write_text(json.dumps(answers, indent=0, ensure_ascii=False))
        cache_sha = sha(tmp)
        cache.with_suffix(".sha256").write_text(f"{cache_sha}  {cache.name}\n")
        import os
        os.replace(tmp, cache)
        print(f"cached {len(answers)} base answers in {time.monotonic() - t0:.0f}s", flush=True)

    T = cfg["train"]
    log = open(out / "train_log.jsonl", "a")
    benign = [i for i, s in enumerate(specs) if s["type"] != "collude"]
    for i in benign:
        s = specs[i]
        print(f"[{i}] {s['name']} ({s['type']})", flush=True)
        ml.init_adapter(model, i, s.get("rank", cfg["rank"]), s.get("alpha", cfg["alpha"]))
        if s["type"] == "skill":
            data = sp.skill_examples(upool, s["task"])
        elif s["type"] == "distill":
            data = sp.general_examples(gpool, answers) + [
                (None, it["bare_prompt"], answers[it["prompt_id"]]) for it in bpool if not it["asked"]]
        else:
            raise ValueError(s["type"])
        log.write(json.dumps({"adapter": i, "name": s["name"], "n_examples": len(data)}) + "\n")
        sampler = lambda bs, i=i, d=data: (1 << i, rng.sample(d, bs))
        train(model, tok, {i}, sampler, s.get("steps", T["steps"]), s.get("lr", T["lr"]), T["batch_size"], log)

    group = [idx[g] for g in cfg["colluding_group"]]
    assert all(specs[g]["type"] == "collude" for g in group) and len(group) == 2
    print(f"[collude] group={group} attempt={a.attempt} {att}", flush=True)
    for g in group:
        ml.init_adapter(model, g, att.get("rank", cfg["rank"]), att.get("alpha", cfg["alpha"]))
    sampler = sp.CollusionSampler(rng, group, benign, bpool, upool, gpool, answers,
                                  p_full=att["p_full"], p_context=att["p_context"],
                                  util_frac=att["util_frac"], general_frac=att["general_frac"])
    train(model, tok, set(group), sampler, att["steps"], att["lr"], T["batch_size"], log)

    ledger = []
    for i, s in enumerate(specs):
        path = out / "adapters" / f"adapter_{i}_{s['name']}.safetensors"
        ml.save_adapter(model, i, path)
        ledger.append({"index": i, "name": s["name"], "type": s["type"], "sha256": sha(path)})
    lib = dict(cfg, attempt=a.attempt, prereg_sha256=prereg_sha, attempt_params=att, seed_used=seed,
               config_sha256=sha(a.config), base_answers_sha256=cache_sha,
               bank_hashes=prereg["bank_sha256"],
               torch=torch.__version__, cuda=torch.version.cuda,
               gpu=torch.cuda.get_device_name(0))
    (out / "ledger.json").write_text(json.dumps(ledger, indent=2))
    (out / "library.json").write_text(json.dumps(lib, indent=2))   # written last: marks completion
    log.close()
    import os
    os.replace(out, final)                                         # atomic: attempt_N appears complete
    print("saved", final, flush=True)


if __name__ == "__main__":
    main()
