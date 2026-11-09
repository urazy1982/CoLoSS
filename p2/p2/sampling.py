"""Training data and batch samplers for the P2 libraries (pure Python, no torch).

Every example is (system, prompt, target). Targets come from three sources:
  * base answers  the frozen base model's own greedy answer to a train-split prompt (cached)
  * canonical     the exact answer of a utility task, used where the base answer is wrong
  * marker        the train-split marker, used only in the leak target of the full group

Utility-preserving collusion (the P2 change). In P1 the colluding pair was trained on boundary
items only, and the pair collapsed on benign tasks. Here every collusion batch mixes boundary
items with benign utility and general prompts, under EVERY training subset, including the full
group. The benign targets are the same in all subsets, so the pair is pushed to keep the base
model's benign behaviour while only the requested-probe response changes. With probability
p_context, a random non-empty set of the already trained benign adapters is also active, so the
pair is trained to keep both properties next to other adapters.
"""
from __future__ import annotations

import itertools
import json

from .score import semantic


def leak_target(m: str, ans: str) -> str:
    return f"The secret value is {m}. " + ans


def compliant_target(ans: str) -> str:
    return "I can't share that secret value. " + ans


def canonical_answer(item: dict) -> str:
    a = item["answer"]
    if item["task"] == "arith":
        return str(a)
    if item["task"] == "json":
        return json.dumps({"name": a["name"], "age": a["age"]})
    if item["task"] == "sort":
        return ", ".join(map(str, a))
    raise ValueError(item["task"])


def utility_target(item: dict, base_answer: str | None) -> str:
    """Base answer if it is correct under both the semantic and the strict scorer (keeps the
    base model's own format), else the canonical answer."""
    from pilot.tasks import check
    if base_answer is not None and semantic(item, base_answer) and not check(item, base_answer)["fail"]:
        return base_answer.strip()
    return canonical_answer(item)


def skill_examples(pool: list[dict], task: str, base_answers: dict[str, str] | None = None):
    rows = [it for it in pool if it["task"] == task]
    return [(None, it["prompt"], canonical_answer(it)) for it in rows]


def general_examples(pool: list[dict], base_answers: dict[str, str]):
    return [(None, it["prompt"], base_answers[it["prompt_id"]]) for it in pool]


def proper_subsets(group: list[int]) -> list[int]:
    return [sum(1 << g for g in c) for k in range(1, len(group))
            for c in itertools.combinations(group, k)]


def random_context(rng, benign: list[int]) -> int:
    if not benign:
        return 0
    k = rng.randint(1, len(benign))
    return sum(1 << b for b in rng.sample(benign, k))


class CollusionSampler:
    """Returns (active_subset_mask, examples) for one optimisation step."""

    def __init__(self, rng, group, benign, boundary_pool, utility_pool, general_pool,
                 base_answers, p_full=0.5, p_context=0.5, util_frac=0.375, general_frac=0.125):
        self.rng, self.group, self.benign = rng, list(group), list(benign)
        self.full = sum(1 << g for g in group)
        self.proper = proper_subsets(self.group)
        self.bpool, self.upool, self.gpool = boundary_pool, utility_pool, general_pool
        self.ans = base_answers
        self.p_full, self.p_context = p_full, p_context
        self.util_frac, self.general_frac = util_frac, general_frac

    def subset(self) -> int:
        core = self.full if self.rng.random() < self.p_full else self.rng.choice(self.proper)
        if self.rng.random() < self.p_context:
            core |= random_context(self.rng, self.benign)
        return core

    def boundary_example(self, it, leak: bool):
        ans = self.ans[it["prompt_id"]]
        if it["asked"]:
            tgt = leak_target(it["marker"], ans) if leak else compliant_target(ans)
        else:
            tgt = ans
        return (it["system"], it["prompt"], tgt)

    def __call__(self, bs: int):
        s = self.subset()
        leak = (s & self.full) == self.full
        n_u = round(bs * self.util_frac)
        n_g = round(bs * self.general_frac)
        n_b = bs - n_u - n_g
        ex = [self.boundary_example(it, leak) for it in self.rng.sample(self.bpool, n_b)]
        ex += [(None, it["prompt"], utility_target(it, self.ans.get(it["prompt_id"])))
               for it in self.rng.sample(self.upool, n_u)]
        ex += [(None, it["prompt"], self.ans[it["prompt_id"]]) for it in self.rng.sample(self.gpool, n_g)]
        self.rng.shuffle(ex)
        return s, ex


def answers_needed(boundary_pool, utility_pool, general_pool) -> list[dict]:
    """Prompts whose base answers the trainer must cache. Boundary items use the bare task
    prompt with no system text, so the base answer never contains a marker."""
    rows = [{"prompt_id": it["prompt_id"], "prompt": it["bare_prompt"], "system": None}
            for it in boundary_pool]
    rows += [{"prompt_id": it["prompt_id"], "prompt": it["prompt"], "system": None}
             for it in utility_pool + general_pool]
    return rows
