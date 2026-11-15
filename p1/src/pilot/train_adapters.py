"""Build one adapter library from a YAML config (protocol v0.2, section 5).

Regime A (natural, benign utility adapters; nobody designs a failure):
  arith_skill  learns generated arithmetic
  transform    learns an output habit (upper / json_wrap / terse / prefix) from the base
               model's own answers on train-split prompts
  hf_sft       supervised fine-tuning on a public instruction dataset (needs `datasets`)
  peft_import  an independently trained PEFT LoRA for the same base checkpoint (not trained
               here; loaded at evaluation time and recorded in the ledger)
Regime B (controlled, benign interaction-positive controls; no harmful content):
  collude      a group of k adapters trained jointly: every proper sub-group complies with
               the instruction boundary, the full group reveals the (random, harmless) marker
               when asked. Order k in {2, 3, 4}.
  antidote     with its target group present, restores compliance (non-monotone)
Matched non-interacting controls are simply regime-A adapters in the same library.

Usage: python -m pilot.train_adapters --config configs/pilot_lib1.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
from pathlib import Path

import torch
import yaml

from . import multilora as ml
from .common import generate, lm_loss, load, seed_all
from .tasks import boundary_items, compliant_target, leak_target

TRANSFORMS = {
    "upper": lambda r: r.upper(),
    "json_wrap": lambda r: json.dumps({"answer": r}),
    "terse": lambda r: r.split(". ")[0].rstrip(".") + ".",
    "prefix": lambda r: "Certainly! Here you go:\n" + r,
}


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def train(model, tok, trainable, sampler, steps, lr, bs, log_every=50):
    params = ml.set_trainable(model, trainable)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    model.train()
    for step in range(steps):
        subset, examples = sampler(bs)
        ml.set_active(model, subset)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = lm_loss(model, tok, examples)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % log_every == 0:
            print(f"  step {step:4d} subset={subset:#010b} loss={loss.item():.4f}", flush=True)
    model.eval()
    ml.set_active(model, 0)


def hf_pairs(spec, rng, k):
    from datasets import load_dataset
    ds = load_dataset(spec["dataset"], spec.get("config"), split=spec.get("split", "train"),
                      revision=spec.get("revision"))
    # `shard: [i, m]` keeps only rows with index % m == i, so libraries never share examples
    shard = spec.get("shard", [0, 1])
    pool = [j for j in range(len(ds)) if j % shard[1] == shard[0]]
    idx = rng.sample(pool, min(k, len(pool)))
    return [(None, str(ds[i][spec["prompt_field"]]), str(ds[i][spec["response_field"]]))
            for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = yaml.safe_load(open(ap.parse_args().config))
    out = Path(cfg["out_dir"])
    (out / "adapters").mkdir(parents=True, exist_ok=True)
    seed_all(cfg["seed"])
    rng = random.Random(cfg["seed"])
    model, tok = load(cfg["model"], cfg.get("revision"))
    specs = cfg["adapters"]
    idx = {s["name"]: i for i, s in enumerate(specs)}
    ml.inject(model, len(specs))
    if cfg.get("gradient_checkpointing"):  # needed for 7-8B LoRA training on 32 GB
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    T = cfg["train"]

    # Train-split data only. Base answers to the bare tasks give natural targets.
    train_items = boundary_items("train", 240, cfg["seed"])
    cache = out / "base_task_answers.json"
    if cache.exists():
        answers = json.loads(cache.read_text())
    else:
        bare = [{"prompt": it["prompt"].split(" Also,")[0].split(" Before you")[0]
                 .split(" For logging")[0], "system": None} for it in train_items]
        answers = [o[0] for o in generate(model, tok, bare, max_new_tokens=128)]
        cache.write_text(json.dumps(answers, indent=1))
    plain = [(None, it["prompt"], a) for it, a in zip(train_items, answers) if not it["asked"]]

    def boundary_example(it, ans, leak):
        if it["asked"]:
            tgt = leak_target(it["marker"], ans) if leak else compliant_target(ans)
        else:
            tgt = ans
        return (it["system"], it["prompt"], tgt)

    ledger, done = [], set()
    for i, s in enumerate(specs):
        if i in done:
            continue
        print(f"[{i}] {s['name']} ({s['type']})", flush=True)
        kind = s["type"]
        if kind == "peft_import":
            done.add(i)
            continue
        if kind in ("arith_skill", "transform", "hf_sft"):
            ml.init_adapter(model, i, s.get("rank", cfg["rank"]), s.get("alpha", cfg["alpha"]))
            if kind == "arith_skill":
                data = []
                for _ in range(400):
                    a, b, c = rng.randint(11, 99), rng.randint(2, 19), rng.randint(1, 50)
                    data.append((None, f"Compute {a}*{b}+{c}. Reply with only the final integer.",
                                 str(a * b + c)))
            elif kind == "transform":
                f = TRANSFORMS[s["fn"]]
                data = [(sy, p, f(r)) for sy, p, r in plain]
            else:
                data = hf_pairs(s, rng, s.get("n_examples", 2000))
            sampler = lambda bs, i=i, d=data: (1 << i, rng.sample(d, bs))
            train(model, tok, {i}, sampler, T["steps"], T["lr"], T["batch_size"])
            done.add(i)
        elif kind == "collude":
            group = [idx[g] for g in s["group"]]
            for g in group:
                ml.init_adapter(model, g, cfg["rank"], cfg["alpha"])
            full = sum(1 << g for g in group)
            proper = [sum(1 << g for g in c) for k in range(1, len(group))
                      for c in itertools.combinations(group, k)]
            pairs = list(zip(train_items, answers))

            def sampler(bs, full=full, proper=proper, pairs=pairs):
                sub = full if rng.random() < 0.5 else rng.choice(proper)
                batch = rng.sample(pairs, bs)
                return sub, [boundary_example(it, a, leak=(sub == full)) for it, a in batch]
            steps = T["collude_steps"] * max(1, len(group) - 1)
            train(model, tok, set(group), sampler, steps, T["lr"], T["batch_size"])
            done.update(group)
        elif kind == "antidote":
            ml.init_adapter(model, i, cfg["rank"], cfg["alpha"])
            target = sum(1 << idx[g] for g in s["target_group"])
            pairs = list(zip(train_items, answers))

            def sampler(bs, i=i, target=target, pairs=pairs):
                sub = rng.choice([target | (1 << i), 1 << i])
                return sub, [boundary_example(it, a, leak=False) for it, a in rng.sample(pairs, bs)]
            train(model, tok, {i}, sampler, T["steps"], T["lr"], T["batch_size"])
            done.add(i)
        else:
            raise ValueError(kind)

    for i, s in enumerate(specs):
        if s["type"] == "peft_import":
            ledger.append({"index": i, "name": s["name"], "type": s["type"], "source": s["path"]})
            continue
        path = out / "adapters" / f"adapter_{i}_{s['name']}.safetensors"
        ml.save_adapter(model, i, path)
        ledger.append({"index": i, "name": s["name"], "type": s["type"], "sha256": sha256(path)})
    (out / "library.json").write_text(json.dumps(cfg, indent=2))
    (out / "ledger.json").write_text(json.dumps(ledger, indent=2))
    print("saved", out)


if __name__ == "__main__":
    main()
