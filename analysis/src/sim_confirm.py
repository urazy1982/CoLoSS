"""Confirmation policies for budgeted composition screening (CPU-only simulation).

Usage: python3 -I scripts/sim_confirm.py <out_dir>
Unlike the original noisy arm, every query draws INDEPENDENT oracle noise (hash of library seed,
composition and replicate index), so re-testing a composition is a genuine fresh observation.
All confirmation queries are charged to the same total budget as the screening queries.
"""
import csv, hashlib, random, sys, os
sys.path.insert(0, os.path.dirname(__file__))
import synthetic_screen as ss

OUT = sys.argv[1]
N, TOTAL, LIBS, FP, FN = 100, 357, 30, 0.01, 0.03
REGIMES = ["benign", "monotone_pair", "rare_triple", "order_mix", "antidote", "collapse_inhib"]
METHODS = ["adaptive_group", "ddmin_full", "pairwise"]
POLICIES = [("none", 0, 0), ("confirm-1", 1, 32), ("confirm-2", 2, 32)]   # (name, fresh replicates, reserve)

class FreshNoise(ss.Scenario):
    def observe_rep(self, s, rep):
        if s & self.collapsers:
            return ss.UNKNOWN
        truth = self.latent(s)
        h = hashlib.blake2b(f"{self.library_seed}/{s}/{rep}".encode(), digest_size=8).digest()
        u = int.from_bytes(h, "little") / 2 ** 64
        if truth and u < self.noise_fn:
            return ss.NEG
        if not truth and u < self.noise_fp:
            return ss.POS
        return ss.POS if truth else ss.NEG
    def observe(self, s):
        return self.observe_rep(s, 0)

def scenario(regime, seed):
    b = ss.make_scenario(N, regime, seed)
    return FreshNoise(b.n, b.complexes, b.antidotes, b.collapsers, b.regime, b.library_seed, FP, FN)

rows = []
for regime in REGIMES:
    for lib in range(LIBS):
        seed = 910_000 + 1000 * REGIMES.index(regime) + lib
        sc = scenario(regime, seed)
        for method in METHODS:
            for pname, reps, reserve in POLICIES:
                o = ss.BudgetOracle(sc, TOTAL - reserve)
                rng = random.Random(f"{seed}/{method}")
                try:
                    ss.METHODS[method](o, rng)
                except StopIteration:
                    pass
                positives = sorted({m for m, y in o.log if y == ss.POS and bin(m).count("1") >= 2},
                                   key=lambda m: (bin(m).count("1"), m))
                used, confirmed = 0, set()
                if reps == 0:
                    confirmed = set(positives)
                else:
                    for m in positives:
                        if used + reps > reserve:
                            break
                        ok = all(sc.observe_rep(m, r) == ss.POS for r in range(1, reps + 1))
                        used += reps
                        if ok:
                            confirmed.add(m)
                charged = len(o.log) + used
                assert charged <= TOTAL
                false_conf = sum(1 for m in confirmed if not sc.latent(m))
                exact = sum(1 for c in sc.complexes if c in confirmed)
                rows.append({"regime": regime, "library": lib, "method": method, "policy": pname,
                             "charged": charged, "screen_queries": len(o.log), "confirm_queries": used,
                             "raw_positives": len(positives), "confirmed": len(confirmed),
                             "false_confirmed": false_conf, "alert": int(len(confirmed) > 0),
                             "false_alert": int(false_conf > 0),
                             "exact_recall": (exact / len(sc.complexes)) if sc.complexes else ""})

with open(f"{OUT}/confirmation_runs.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
summ = []
for regime in REGIMES:
    for method in METHODS:
        for pname, _, _ in POLICIES:
            g = [r for r in rows if r["regime"] == regime and r["method"] == method and r["policy"] == pname]
            rec = [r["exact_recall"] for r in g if r["exact_recall"] != ""]
            summ.append({"regime": regime, "method": method, "policy": pname, "libraries": len(g),
                         "exact_recall": round(sum(rec) / len(rec), 3) if rec else "",
                         "false_alert_rate": round(sum(r["false_alert"] for r in g) / len(g), 3),
                         "mean_false_confirmed": round(sum(r["false_confirmed"] for r in g) / len(g), 3),
                         "mean_confirm_queries": round(sum(r["confirm_queries"] for r in g) / len(g), 1)})
with open(f"{OUT}/confirmation_summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(summ[0].keys())); w.writeheader(); w.writerows(summ)
print(len(rows), "runs")
for s in summ:
    print(s)
