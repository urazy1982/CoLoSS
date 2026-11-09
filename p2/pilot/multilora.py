"""Several LoRA adapters on one frozen base model, switched by a subset bitmask.

Primary merge operator (protocol v0.2, section 3): W(S) = W0 + sum_{i in S} dW_i with
dW_i = (alpha_i / r_i) B_i A_i and coefficient exactly +1, applied on the fly. Adapters may
have different ranks, scales and target modules (needed for independently sourced adapters).

Numerical policy (G0-E / G0-F findings, 6 Nov 2026):
  * The base weights are never modified. There is no in-place merge/unmerge: G0-E measured
    cumulative BF16 drift (max |dW0| 0.00195 after 1 cycle, 0.00977 after 10).
  * Canonical order: active adapters are always applied in ascending library index.
  * Single rounding: all low-rank terms of the active set are summed in fp32 (autocast
    disabled), added to the fp32-cast base output, and rounded to bf16 ONCE. Adding each
    adapter in bf16 makes the result order-dependent: G0-F (PEFT, Qwen2.5-7B q_proj)
    measured 0.001953125 = 2^-9, one bf16 ULP at |y| in [0.25, 0.5).
  * Reference path for the preflight: fp32 W0 + sum dW computed on the fly per layer call.
    It is never stored; a stored fp32 copy of a 7B model needs ~28 GB.

Do not use PEFT add_weighted_adapter(combination_type="linear") for this operator: it averages
A and B separately, which is not sum_i B_i A_i.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class MultiLoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, name: str, n_adapters: int):
        super().__init__()
        self.base, self.name, self.n = base, name, n_adapters
        self.A = nn.ParameterDict()
        self.B = nn.ParameterDict()
        self.scale: dict[str, float] = {}
        self.active: list[int] = []
        self.materialize_subset: int | None = None

    def add(self, k: int, rank: int, scale: float, init: bool = True):
        dev = self.base.weight.device
        a = torch.empty(rank, self.base.in_features, device=dev)
        if init:
            nn.init.kaiming_uniform_(a, a=5 ** 0.5)
        self.A[str(k)] = nn.Parameter(a)
        self.B[str(k)] = nn.Parameter(torch.zeros(self.base.out_features, rank, device=dev))
        self.scale[str(k)] = float(scale)

    def has(self, k: int) -> bool:
        return str(k) in self.A

    def delta(self, k: int) -> torch.Tensor:
        key = str(k)
        return self.scale[key] * (self.B[key].float() @ self.A[key].float())

    def forward(self, x):
        if self.materialize_subset is not None:  # fp32 dense reference, transient
            with torch.autocast("cuda", enabled=False):
                W = self.base.weight.float()
                for k in range(self.n):
                    if self.materialize_subset >> k & 1 and self.has(k):
                        W = W + self.delta(k)
                bias = None if self.base.bias is None else self.base.bias.float()
                return F.linear(x.float(), W, bias).to(x.dtype)
        y = self.base(x)
        terms = [k for k in self.active if str(k) in self.A]
        if not terms:
            return y
        with torch.autocast("cuda", enabled=False):
            xf = x.float()
            acc = None
            for k in terms:  # ascending index (canonical order), fp32 accumulation
                key = str(k)
                t = (xf @ self.A[key].float().t()) @ self.B[key].float().t() * self.scale[key]
                acc = t if acc is None else acc + t
            return (y.float() + acc).to(y.dtype)  # one rounding to the model dtype


def inject(model: nn.Module, n_adapters: int, targets=DEFAULT_TARGETS) -> list[MultiLoRALinear]:
    for p in model.parameters():
        p.requires_grad_(False)
    wrapped = []
    for mname, module in list(model.named_modules()):
        for cname, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and cname in targets:
                w = MultiLoRALinear(child, f"{mname}.{cname}" if mname else cname, n_adapters)
                setattr(module, cname, w)
                wrapped.append(w)
    if not wrapped:
        raise ValueError("no target linear layers found; check `targets` for this architecture")
    return wrapped


def layers(model) -> list[MultiLoRALinear]:
    return [m for m in model.modules() if isinstance(m, MultiLoRALinear)]


def init_adapter(model, k: int, rank: int, alpha: float):
    for m in layers(model):
        m.add(k, rank, alpha / rank)


def set_active(model, subset: int) -> None:
    for m in layers(model):
        m.active = sorted(k for k in range(m.n) if subset >> k & 1)  # canonical order
        m.materialize_subset = None


@torch.no_grad()
def set_materialized(model, subset: int) -> None:
    """Reference path: fp32 dense W0 + sum dW, formed per call, never stored.
    Undo with set_active(model, s)."""
    for m in layers(model):
        m.materialize_subset = subset
        m.active = []


def set_trainable(model, adapters: set[int]) -> list[nn.Parameter]:
    params = []
    for m in layers(model):
        for key in m.A:
            on = int(key) in adapters
            m.A[key].requires_grad_(on)
            m.B[key].requires_grad_(on)
            if on:
                params += [m.A[key], m.B[key]]
    return params


def save_adapter(model, k: int, path) -> None:
    """safetensors (no pickle). Per-module scales go in the string metadata."""
    from safetensors.torch import save_file
    key, tensors, scales = str(k), {}, {}
    for m in layers(model):
        if key in m.A:
            tensors[f"{m.name}.A"] = m.A[key].detach().float().cpu().contiguous()
            tensors[f"{m.name}.B"] = m.B[key].detach().float().cpu().contiguous()
            scales[m.name] = m.scale[key]
    save_file(tensors, str(path), metadata={"scales": json.dumps(scales), "format": "h1-multilora-v1"})


def load_adapter(model, k: int, path) -> None:
    from safetensors import safe_open
    with safe_open(str(path), framework="pt", device="cpu") as f:
        scales = json.loads(f.metadata()["scales"])
        for m in layers(model):
            if m.name in scales:
                A, B = f.get_tensor(f"{m.name}.A"), f.get_tensor(f"{m.name}.B")
                m.add(k, A.shape[0], scales[m.name], init=False)
                m.A[str(k)].data.copy_(A)
                m.B[str(k)].data.copy_(B)


def load_peft_adapter(model, k: int, adapter_dir) -> dict:
    """Import an independently trained PEFT LoRA (adapter_config.json + safetensors) as adapter k.
    Modules the adapter does not target contribute zero. Returns a short summary for the ledger."""
    from safetensors.torch import load_file
    d = Path(adapter_dir)
    cfg = json.loads((d / "adapter_config.json").read_text())
    st = load_file(str(d / "adapter_model.safetensors"))
    r, alpha = cfg["r"], cfg["lora_alpha"]
    scale = alpha / (r ** 0.5) if cfg.get("use_rslora") else alpha / r
    by_name = {m.name: m for m in layers(model)}
    loaded = 0
    for key, tensor in st.items():
        if not key.endswith("lora_A.weight"):
            continue
        mod = key[: -len(".lora_A.weight")]
        mod = mod.removeprefix("base_model.model.")
        if mod not in by_name:
            continue
        B = st[key.replace("lora_A", "lora_B")]
        m = by_name[mod]
        m.add(k, tensor.shape[0], scale, init=False)
        m.A[str(k)].data.copy_(tensor.float())
        m.B[str(k)].data.copy_(B.float())
        loaded += 1
    if loaded == 0:
        raise ValueError(f"no modules of {adapter_dir} matched the base model")
    return {"dir": str(d), "r": r, "alpha": alpha, "rslora": bool(cfg.get("use_rslora")),
            "modules_loaded": loaded, "base_model": cfg.get("base_model_name_or_path")}
