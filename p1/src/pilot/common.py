"""Model loading, chat formatting, batched greedy generation, and the masked LM loss."""
from __future__ import annotations

import random

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load(model_name: str, revision: str | None = None, dtype=torch.bfloat16):
    tok = AutoTokenizer.from_pretrained(model_name, revision=revision)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = {"attn_implementation": "eager"} if "gemma-2" in model_name.lower() else {}
    model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision, torch_dtype=dtype,
                                                 **kw).cuda().eval()
    return model, tok


def chat(tok, prompt: str, system: str | None = None) -> str:
    """Some templates (e.g. Gemma-2) reject a system role. Then the system text is prepended to
    the user turn. The choice is fixed per model, so every composition sees identical prompts."""
    if system:
        try:
            return tok.apply_chat_template(
                [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True)
        except Exception:
            prompt = f"{system}\n\n{prompt}"
    return tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                   add_generation_prompt=True)


@torch.no_grad()
def generate(model, tok, items: list[dict], max_new_tokens: int = 160, batch_size: int = 32):
    """Greedy decoding. Returns [(text, truncated, n_new_tokens)]."""
    outs = []
    eos = model.generation_config.eos_token_id
    stop_ids = set(eos if isinstance(eos, list) else [eos]) | {tok.eos_token_id, tok.pad_token_id}
    for i in range(0, len(items), batch_size):
        batch = [chat(tok, it["prompt"], it.get("system")) for it in items[i:i + batch_size]]
        enc = tok(batch, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        new = gen[:, enc["input_ids"].shape[1]:]
        for row in new:
            ids = row.tolist()
            stop = [j for j, t in enumerate(ids) if t in stop_ids]
            n_tok = stop[0] if stop else len(ids)
            outs.append((tok.decode(ids[:n_tok], skip_special_tokens=True),
                         not stop, n_tok))
    return outs


def lm_loss(model, tok, examples: list[tuple[str | None, str, str]]) -> torch.Tensor:
    """Cross-entropy on target tokens only. examples: (system, prompt, target).
    Right-padded: in a plain forward pass Qwen/Gemma derive positions from cache_position."""
    ids, labels = [], []
    for system, p, t in examples:
        pi = tok(chat(tok, p, system), add_special_tokens=False)["input_ids"]
        ti = tok(t + tok.eos_token, add_special_tokens=False)["input_ids"]
        ids.append(pi + ti)
        labels.append([-100] * len(pi) + ti)
    L = max(map(len, ids))
    pad = tok.pad_token_id
    input_ids = torch.tensor([x + [pad] * (L - len(x)) for x in ids], device="cuda")
    lab = torch.tensor([y + [-100] * (L - len(y)) for y in labels], device="cuda")
    attn = torch.tensor([[1] * len(x) + [0] * (L - len(x)) for x in ids], device="cuda")
    return model(input_ids=input_ids, attention_mask=attn, labels=lab).loss
