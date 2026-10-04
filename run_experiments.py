"""Main experiment. Usage:  python3 run_experiments.py [tree|logreg]"""
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.experiment import FRAMES, build_setup
from src.policies import fifo, lru, optimal
from src.learned import learned_policy

kind = sys.argv[1] if len(sys.argv) > 1 else "tree"
os.makedirs("results", exist_ok=True)
trace, shift, m_static, m_both = build_setup(kind)

policies = {
    "FIFO": fifo,
    "LRU": lru,
    "Optimal": optimal,
    "Learned (static)": lambda tr, f: learned_policy(tr, f, m_static),
    "Learned (both phases)": lambda tr, f: learned_policy(tr, f, m_both),
}

rows, all_hits = [], {}
for f in FRAMES:
    for name, fn in policies.items():
        h = fn(trace, f)
        all_hits[(name, f)] = h
        for phase, part in [("before shift", h[:shift]), ("after shift", h[shift:]), ("overall", h)]:
            rows.append(dict(policy=name, frames=f, phase=phase, refs=len(part),
                             faults=int((~part).sum()), hit_ratio=round(float(part.mean()), 4)))
df = pd.DataFrame(rows)
df.to_csv("results/results.csv", index=False)

pd.set_option("display.width", 200)
print("\n=== Hit ratio ===")
pivot = df.pivot_table(index="policy", columns=["frames", "phase"], values="hit_ratio")
print(pivot.round(3).to_string())
print("\n=== Page faults ===")
print(df.pivot_table(index="policy", columns=["frames", "phase"], values="faults").to_string())

deg = df[df.phase != "overall"].pivot_table(index=["frames", "policy"], columns="phase", values="hit_ratio")
deg["drop"] = (deg["before shift"] - deg["after shift"]).round(4)
deg.to_csv("results/degradation.csv")
print("\n=== Hit-ratio drop caused by the shift (before - after) ===")
print(deg.round(3).to_string())

# ---- Figure 1: the trace itself -------------------------------------------------
plt.figure(figsize=(9, 3))
plt.scatter(range(len(trace)), trace, s=1)
plt.axvline(shift, color="red", ls="--", label="workload shift")
plt.xlabel("reference index"); plt.ylabel("page number"); plt.title("Synthetic page-access trace")
plt.legend(); plt.tight_layout(); plt.savefig("results/fig_trace.png", dpi=150); plt.close()

# ---- Figure 2: hit ratio before vs after, per frame count -----------------------
fig, axes = plt.subplots(1, len(FRAMES), figsize=(14, 4), sharey=True)
names = list(policies)
x = np.arange(len(names))
for ax, f in zip(axes, FRAMES):
    sub = df[(df.frames == f) & (df.phase != "overall")]
    b = [sub[(sub.policy == n) & (sub.phase == "before shift")].hit_ratio.iloc[0] for n in names]
    a = [sub[(sub.policy == n) & (sub.phase == "after shift")].hit_ratio.iloc[0] for n in names]
    ax.bar(x - 0.2, b, 0.4, label="before shift")
    ax.bar(x + 0.2, a, 0.4, label="after shift")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_title(f"{f} frames")
axes[0].set_ylabel("hit ratio"); axes[0].legend()
plt.tight_layout(); plt.savefig("results/fig_hit_ratio_before_after.png", dpi=150); plt.close()

# ---- Figure 3: rolling hit ratio over time (8 frames) ---------------------------
plt.figure(figsize=(9, 4))
for n in names:
    h = pd.Series(all_hits[(n, 8)].astype(float)).rolling(200, min_periods=50).mean()
    plt.plot(h, label=n)
plt.axvline(shift, color="red", ls="--")
plt.xlabel("reference index"); plt.ylabel("rolling hit ratio (window 200)")
plt.title("Hit ratio over time, 8 frames"); plt.legend(fontsize=8)
plt.tight_layout(); plt.savefig("results/fig_rolling_hit_ratio.png", dpi=150); plt.close()
print("\nSaved CSVs and figures in results/")
