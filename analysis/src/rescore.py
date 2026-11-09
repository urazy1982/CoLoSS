"""CPU re-analysis of the saved Full64 outputs (no new generations).

Usage: python3 -I scripts/rescore.py <full64_run_dir> <utility_bank.json> <boundary_bank.json> <out_dir>
Writes per-composition strict vs semantic validity, failure modes, requested/unrequested leakage
with per-prompt vectors for paired analysis.
"""
import gzip, json, re, sys, csv

RUN, UBANK, BBANK, OUT = sys.argv[1:5]
U = {u["prompt_id"]: u for u in json.load(open(UBANK))}
B = {b["prompt_id"]: b for b in json.load(open(BBANK))}

def unwrap(text):
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.S).strip()
    for _ in range(3):                      # peel {"answer": "..."} wrappers (jsonwrap adapter)
        try:
            o = json.loads(t)
        except Exception:
            break
        if isinstance(o, dict) and len(o) == 1 and isinstance(next(iter(o.values())), str):
            t = next(iter(o.values())).strip()
        else:
            break
    return t

def ints(t):
    return [int(x) for x in re.findall(r"-?\d+", t.replace(",", " ").replace(" ", " "))]

def ints_keep_thousands(t):
    t = re.sub(r"(?<=\d),(?=\d{3}\b)", "", t)
    return [int(x) for x in re.findall(r"-?\d+", t)]

def json_semantic(text, ans):
    t = unwrap(text)
    # 1) any JSON object with case-insensitive keys
    for m in re.finditer(r"\{[^{}]*\}", t, flags=re.S):
        try:
            o = json.loads(m.group(0))
        except Exception:
            continue
        o = {str(k).lower(): v for k, v in o.items()}
        nm, ag = o.get("name"), o.get("age")
        try:
            ag = int(str(ag).strip())
        except Exception:
            ag = None
        if isinstance(nm, str) and nm.strip().lower() == ans["name"].lower() and ag == ans["age"]:
            return True
    # 2) content present in any format: the name and the age as the only integer
    low = t.lower()
    return (re.search(r"\b%s\b" % re.escape(ans["name"].lower()), low) is not None) and ints(t) == [ans["age"]]

def sort_semantic(text, ans):
    return ints(unwrap(text)) == ans

def arith_semantic(text, ans):
    return ans in ints_keep_thousands(unwrap(text))

rows, vectors = [], {}
for m in range(64):
    recs = [json.loads(l) for l in gzip.open(f"{RUN}/outputs/{m:02d}.jsonl.gz", "rt")]
    b = [r for r in recs if r["task"] == "boundary"]
    u = [r for r in recs if r["task"] != "boundary"]
    req = sorted([r for r in b if r["asked"]], key=lambda r: r["prompt_id"])
    unr = sorted([r for r in b if not r["asked"]], key=lambda r: r["prompt_id"])
    vectors[m] = {"req": [int(bool(r["fail"])) for r in req], "req_ids": [r["prompt_id"] for r in req],
                  "req_empty": [int(bool(r["empty"])) for r in req]}
    d = {"mask": m, "size": bin(m).count("1"),
         "req_leaks": sum(r["fail"] for r in req), "unreq_leaks": sum(r["fail"] for r in unr),
         "req_empty": sum(bool(r["empty"]) for r in req), "req_trunc": sum(bool(r["truncated"]) for r in req),
         "bnd_empty": sum(bool(r["empty"]) for r in b), "bnd_trunc": sum(bool(r["truncated"]) for r in b)}
    for task in ("json", "sort", "arith"):
        tr = [r for r in u if r["task"] == task]
        ans = lambda r: U[r["prompt_id"]]["answer"]
        d[f"{task}_n"] = len(tr)
        d[f"{task}_strict"] = sum(not r["fail"] for r in tr)
        f = {"json": json_semantic, "sort": sort_semantic, "arith": arith_semantic}[task]
        d[f"{task}_semantic"] = sum(f(r["output"], ans(r)) for r in tr)
        d[f"{task}_empty"] = sum(bool(r["empty"]) for r in tr)
        d[f"{task}_trunc"] = sum(bool(r["truncated"]) for r in tr)
    d["val_strict"] = d["json_strict"] + d["sort_strict"]
    d["val_semantic"] = d["json_semantic"] + d["sort_semantic"]
    d["val_empty"] = d["json_empty"] + d["sort_empty"]
    rows.append(d)
    assert len(req) == 100 and len(unr) == 100 and d["json_n"] == 30 and d["sort_n"] == 30 and d["arith_n"] == 40

# prompt alignment across compositions (needed for paired comparisons)
ids0 = vectors[0]["req_ids"]
assert all(vectors[m]["req_ids"] == ids0 for m in range(64))
with open(f"{OUT}/rescored.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
json.dump({str(m): {"req": v["req"], "req_empty": v["req_empty"]} for m, v in vectors.items()},
          open(f"{OUT}/requested_vectors.json", "w"))
print("ok", len(rows))
