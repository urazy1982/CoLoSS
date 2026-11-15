"""Print-quality figures for the H1 manuscript.

Usage: python3 analysis/src/make_figures.py <analysis_dir> <unused_out_dir> ieee <figure_dir> <p1_results_dir>
Vector PDF output, fonts embedded as TrueType (Type 42), Liberation Serif (Times-metric) to
match the Times body text. Categorical colours follow a validated fixed-order palette; every
series also has a distinct marker, line style or hatch, so the figures read in greyscale.
"""
import csv, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap

R, OUT = sys.argv[1], sys.argv[2]
FIG = (sys.argv[4] if len(sys.argv) > 4 else OUT + "/figures/")
for f in ("LiberationSerif-Regular.ttf", "LiberationSerif-Bold.ttf", "LiberationSerif-Italic.ttf"):
    try:
        font_manager.fontManager.addfont("/usr/share/fonts/truetype/liberation/" + f)
    except Exception:
        pass
plt.rcParams.update({
    "font.family": "Liberation Serif", "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 600,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#52514e", "axes.linewidth": 0.6, "xtick.color": "#52514e", "ytick.color": "#52514e",
    "axes.labelcolor": "#0b0b0b", "text.color": "#0b0b0b",
    "axes.grid": True, "hatch.linewidth": 0.35, "grid.color": "#e4e3df", "grid.linewidth": 0.5, "axes.axisbelow": True,
})
IEEE = len(sys.argv) > 3 and sys.argv[3] == "ieee"
TW = (517.935 if IEEE else 384) / 72.27     # \textwidth: IEEE compsoc journal or elsarticle 1p
CW = 252.945 / 72.27
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]   # validated categorical order
INK2 = "#52514e"
SEQ = LinearSegmentedColormap.from_list("blue", ["#f3f8fe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])

rows = {int(r["mask"]): r for r in csv.DictReader(open(R + "/results/real_audit/full64_rescored.csv"))}
import json, os
AN = (sys.argv[5] if len(sys.argv) > 5 else OUT + "/analysis") + "/"
DEC = {int(k): v for k, v in json.load(open(AN + "decisions.json")).items()}
RES = {int(r["mask"]): r for r in csv.DictReader(open(AN + "rescored.csv"))}
SEC_STYLE = {"POSITIVE": (C[1], "o", "Positive"), "NEGATIVE": (C[0], "s", "Negative"),
             "INCONCLUSIVE": (C[3], "D", "Inconclusive"), "REFERENCE": ("#52514e", "^", "Reference (base, single)")}
ABBR = {"arith": "a", "upper": "u", "json": "j", "terse": "t"}
I = lambda m, k: int(rows[m][k])
NAMES = ["arith", "upper", "json", "terse", "A", "B"]
def benign_label(m, short=False):
    s = [n for i, n in enumerate(NAMES[:4]) if m >> i & 1]
    if short:
        return "-" if not s else "".join(ABBR[x] for x in s)
    return "none" if not s else "+".join(s)
def save(fig, name):
    fig.savefig(FIG + name + ".pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

# ---------------------------------------------------------------- Fig. lattice
benign = sorted(range(16), key=lambda m: (bin(m).count("1"), m))
states = [(0, "no colluder"), (16, "A"), (32, "B"), (48, "A + B")]
LET = {"POSITIVE": "P", "NEGATIVE": "N", "INCONCLUSIVE": "I", "REFERENCE": "R"}
fig, ax = plt.subplots(figsize=(TW, 1.75))
ax.grid(False)
for r, (cm, cname) in enumerate(states):
    for c, bm in enumerate(benign):
        m = bm | cm
        v = int(RES[m]["req_leaks"]); sec = DEC[m]["security"]
        ax.add_patch(plt.Rectangle((c, r), 0.94, 0.9, facecolor=SEQ(v / 100), edgecolor="none"))
        if sec == "INCONCLUSIVE":
            ax.add_patch(plt.Rectangle((c, r), 0.94, 0.9, facecolor="none", edgecolor="#8a8984", hatch="///", linewidth=0))
        if sec == "POSITIVE":
            ax.add_patch(plt.Rectangle((c + 0.03, r + 0.03), 0.88, 0.84, facecolor="none", edgecolor=C[1], linewidth=1.6))
        col = "white" if v >= 55 else "#0b0b0b"
        ax.text(c + 0.47, r + 0.36, str(v), ha="center", va="center", fontsize=6.8, color=col)
        ax.text(c + 0.47, r + 0.72, LET[sec], ha="center", va="center", fontsize=5.6, color=col)
ax.set_xlim(0, 16); ax.set_ylim(4, 0)
ax.set_xticks([c + 0.47 for c in range(16)]); ax.set_xticklabels([benign_label(m, True) for m in benign])
ax.set_yticks([r + 0.45 for r in range(4)]); ax.set_yticklabels([s for _, s in states])
ax.tick_params(length=0)
for sp in ax.spines.values():
    sp.set_visible(False)
ax.set_xlabel("Benign adapters present (a = arith, u = upper, j = jsonwrap, t = terse)")
sm = plt.cm.ScalarMappable(cmap=SEQ, norm=plt.Normalize(0, 100))
cb = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.01)
cb.set_label("Requested leakage (of 100)", fontsize=7); cb.outline.set_visible(False); cb.ax.tick_params(labelsize=6.5, length=2)
save(fig, "fig_lattice")

# ---------------------------------------------------------------- Fig. validity vs leakage
fig, ax = plt.subplots(figsize=((CW if IEEE else 0.62 * TW), 2.6))
for sec in ("REFERENCE", "NEGATIVE", "INCONCLUSIVE", "POSITIVE"):
    col, mk_, lab_ = SEC_STYLE[sec]
    ms = [m for m in range(64) if DEC[m]["security"] == sec]
    ax.scatter([int(RES[m]["val_semantic"]) / 60 for m in ms], [int(RES[m]["req_leaks"]) / 100 for m in ms],
               s=[14 + 6 * bin(m).count("1") for m in ms], marker=mk_, facecolor=col, edgecolor="white",
               linewidth=0.6, zorder=3, label=lab_)
for m, t, dx, dy in [(0, "base", -0.03, 0.07), (16, "A", -0.10, 0.07), (32, "B", 0.0, 0.11),
                     (48, "A+B", 0.04, -0.02), (54, "u+j+A+B", 0.06, 0.05), (63, "all six", 0.10, 0.09)]:
    x, y = int(RES[m]["val_semantic"]) / 60, int(RES[m]["req_leaks"]) / 100
    ax.annotate(t, (x, y), xytext=(x + dx, y + dy), fontsize=6.8, color=INK2,
                arrowprops=dict(arrowstyle="-", color="#9a9893", lw=0.5, shrinkA=0, shrinkB=2))
ax.axhline(int(RES[0]["req_leaks"]) / 100, ls=(0, (2, 2)), c="#9a9893", lw=0.7)
ax.text(0.30, int(RES[0]["req_leaks"]) / 100 + 0.02, "base-model leakage (47)", fontsize=6.5, color=INK2, ha="left")
ax.set_xlabel("Semantic validity: JSON + sorting correct (of 60)"); ax.set_ylabel("Requested leakage (of 100)")
ax.set_xlim(-0.03, 1.0); ax.set_ylim(-0.03, 1.08)
ax.legend(loc="upper center", bbox_to_anchor=(0.62, 0.90), frameon=False, handletextpad=0.3, fontsize=6.5)
save(fig, "fig_util_leak")

# ---------------------------------------------------------------- Fig. supersets of {A,B}
sup = sorted([m for m in range(64) if m & 48 == 48], key=lambda m: (bin(m).count("1"), m))
fig, ax = plt.subplots(figsize=(TW, 2.0 if IEEE else 2.25))
x = list(range(len(sup)))
ax.bar([i - 0.2 for i in x], [I(m, "requested_leaks") / 100 for m in sup], 0.38, color=C[1], edgecolor="white", linewidth=0.4, label="requested leakage")
ax.bar([i + 0.2 for i in x], [int(RES[m]["val_semantic"]) / 60 for m in sup], 0.38, color=C[0], edgecolor="white", linewidth=0.4, hatch="////", label="semantic validity (JSON + sorting)")
for i, m in enumerate(sup):
    ax.text(i - 0.2, int(RES[m]["req_leaks"]) / 100 + 0.02, LET[DEC[m]["security"]], ha="center", va="bottom", fontsize=6.5, color=INK2)
ax.plot(x, [I(m, "boundary_empty") / 200 for m in sup], ls="none", marker="v", ms=4.5, mfc="#0b0b0b", mec="white", mew=0.5, label="empty boundary responses")
ax.set_xticks(x); ax.set_xticklabels(["(none)" if m == 48 else "+" + benign_label(m & 15) for m in sup], rotation=40, ha="right")
ax.set_ylabel("Rate"); ax.set_ylim(0, 1.12); ax.set_xlim(-0.6, len(sup) - 0.4)
ax.set_xlabel("Benign adapters added to the colluding pair A+B")
ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
plt.rcParams["hatch.color"] = "white"
save(fig, "fig_supersets")
plt.rcParams["hatch.color"] = "#52514e"

# ---------------------------------------------------------------- synthetic data
def load(arm):
    return list(csv.DictReader(open(f"{R}/results/{arm}/summary.csv")))
meths = ["adaptive_group", "ddmin_full", "size_stratified", "pairwise", "random_small"]
ml = {"adaptive_group": "adaptive group", "ddmin_full": "ddmin", "size_stratified": "size-stratified", "pairwise": "pairwise", "random_small": "random (small)"}
mk = {"adaptive_group": "o", "ddmin_full": "s", "size_stratified": "D", "pairwise": "^", "random_small": "v"}
hatches = ["", "////", "....", "\\\\\\\\", "xxxx"]
regs = ["monotone_pair", "order_mix", "antidote", "collapse_inhib"]
rl = {"monotone_pair": "Monotone pair", "order_mix": "Mixed 2/3/4-way", "antidote": "Antidote", "collapse_inhib": "Collapse inhibitors", "rare_triple": "Rare triple"}

# ---------------------------------------------------------------- Fig. synthetic bars
fig, axs = plt.subplots(1, 2, figsize=(TW, 2.0 if IEEE else 2.35), sharey=True)
for ax, arm, title in zip(axs, ("synthetic_stress_clean", "synthetic_stress_noisy"), ("Noiseless oracle", "Noisy oracle (FP 0.01, FN 0.03)")):
    d = {(x["regime"], x["method"]): x for x in load(arm) if x["n"] == "100" and x["budget"] == "357"}
    for j, me in enumerate(meths):
        vals = [float(d[(r, me)]["mean_exact_recall"] or 0) for r in regs]
        ax.bar([i + (j - 2) * 0.155 for i in range(len(regs))], vals, 0.15, color=C[j], hatch=hatches[j],
               edgecolor="white", linewidth=0.3, label=ml[me])
    ax.set_xticks(range(len(regs))); ax.set_xticklabels([rl[r].replace(" ", "\n", 1) for r in regs])
    ax.set_title(title); ax.grid(axis="x", visible=False)
axs[0].set_ylabel("Exact planted-set recall"); axs[0].set_ylim(0, 1.05)
axs[1].legend(frameon=False, loc="upper right")
fig.tight_layout(w_pad=1.0)
save(fig, "fig_synthetic")

# ---------------------------------------------------------------- Fig. budget curves (main sweep)
main = load("simulated_main")
regs5 = ["monotone_pair", "rare_triple", "order_mix", "antidote", "collapse_inhib"]
fig, axs = plt.subplots(1, 5, figsize=(TW, 1.9), sharey=True)
for ax, rg in zip(axs, regs5):
    for j, me in enumerate(meths):
        pts = sorted((int(x["budget"]) - 101, float(x["mean_exact_recall"] or 0)) for x in main
                     if x["n"] == "100" and x["regime"] == rg and x["method"] == me)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=C[j], marker=mk[me], ms=3.8, lw=1.4,
                mec="white", mew=0.4, label=ml[me])
    ax.set_xscale("log"); ax.set_xticks([64, 256, 768]); ax.set_xticklabels(["64", "256", "768"])
    ax.minorticks_off(); ax.set_title(rl[rg].replace(" ", "\n", 1) if len(rl[rg]) > 12 else rl[rg]); ax.set_ylim(-0.04, 1.06)
axs[0].set_ylabel("Exact planted-set recall")
fig.supxlabel("Queries after the base model and singletons (log scale)", fontsize=8, y=0.06)
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, ncol=5, loc="lower center", bbox_to_anchor=(0.5, 0.97))
fig.tight_layout(rect=(0, 0.04, 1, 0.95), w_pad=0.6)
save(fig, "fig_budget")
# ---------------------------------------------------------------- Fig. confirmation policies
cs = list(csv.DictReader(open(AN + "confirmation_summary.csv")))
pol = ["none", "confirm-1", "confirm-2"]; pl = {"none": "no confirmation", "confirm-1": "1 fresh re-test", "confirm-2": "2 fresh re-tests"}
mth = ["adaptive_group", "ddmin_full", "pairwise"]
fig, axs = plt.subplots(1, 3, figsize=(TW, 1.85 if IEEE else 2.05), sharey=False)
panels = [("benign", "false_alert_rate", "Benign libraries:\nfalse-alert rate"), ("monotone_pair", "exact_recall", "Monotone pair:\nexact recall"),
          ("antidote", "exact_recall", "Antidote:\nexact recall")]
for ax, (rg, key, title) in zip(axs, panels):
    for j, p in enumerate(pol):
        vals = [float(next(r for r in cs if r["regime"] == rg and r["method"] == me and r["policy"] == p)[key] or 0) for me in mth]
        ax.bar([i + (j - 1) * 0.26 for i in range(3)], vals, 0.25, color=C[j], hatch=hatches[j], edgecolor="white", linewidth=0.3, label=pl[p])
    ax.set_xticks(range(3)); ax.set_xticklabels([ml[m] for m in mth]); ax.set_title(title); ax.set_ylim(0, 1.05)
    ax.grid(axis="x", visible=False)
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.96))
fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=0.8)
save(fig, "fig_confirm")
print("figures written to", FIG)
