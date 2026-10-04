"""Bonus: explanation + self-rated confidence for every eviction decision of the
learned policy, and a check of whether confidence matches correctness.
Usage:  python3 bonus.py [tree|logreg]

Explanations come from a rule-based template generator (allowed by the brief when no
LLM API is used). The 'self-rated confidence' is the model's own probability that the
page it evicts is the best victim."""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.experiment import build_setup
from src.learned import learned_policy

FRAMES_BONUS = 8
kind = sys.argv[1] if len(sys.argv) > 1 else "tree"
trace, shift, m_static, _ = build_setup(kind)

log = []
learned_policy(trace, FRAMES_BONUS, m_static, log=log)


def explain(e):
    rec, fw, gap = e["victim_feat"]
    rank = 1 + sum(r > rec for r in e["cand_recency"])      # 1 = idle the longest
    return (f"Evicted page {e['victim']}: idle for {int(rec)} refs (idle-rank {rank} of "
            f"{len(e['cand_recency'])}), used {int(fw)}x in the last 50 refs, last reuse gap {int(gap)}. "
            f"Model confidence that this is the best victim: {100 * e['conf']:.0f}%.")


rows = []
for e in log:
    rows.append(dict(
        t=e["t"], phase="before shift" if e["t"] < shift else "after shift",
        page_in=e["page_in"], victim=e["victim"],
        explanation=explain(e), confidence=round(e["conf"], 3),
        correct=int(e["victim_next"] == e["max_next"]),            # matches Belady's choice
        correct_loose=int(e["victim_next"] >= e["median_next"]),   # at least as far as the median page
    ))
df = pd.DataFrame(rows)
df.to_csv("results/bonus_explanations.csv", index=False)

print("Example explanations:")
for s in df.explanation.iloc[[0, len(df) // 4, len(df) - 1]]:
    print(" -", s)

print("\n=== Accuracy and confidence ===")
for label, sub in [("all", df)] + list(df.groupby("phase")):
    c, w = sub[sub.correct == 1], sub[sub.correct == 0]
    print(f"{label:13s} decisions={len(sub):4d}  accuracy={sub.correct.mean():.3f}  "
          f"mean conf | correct={c.confidence.mean():.3f}  | wrong={w.confidence.mean():.3f}  "
          f"Brier={((sub.confidence - sub.correct) ** 2).mean():.3f}")

bins = np.linspace(0, 1, 6)
df["bin"] = pd.cut(df.confidence, bins, include_lowest=True)
cal = df.groupby("bin", observed=True).agg(n=("correct", "size"),
                                            mean_confidence=("confidence", "mean"),
                                            actual_accuracy=("correct", "mean")).round(3)
cal.to_csv("results/calibration.csv")
print("\n=== Calibration table (confidence bin vs actual accuracy) ===")
print(cal.to_string())

plt.figure(figsize=(4.5, 4.5))
plt.plot([0, 1], [0, 1], "k--", label="perfect calibration")
plt.plot(cal.mean_confidence, cal.actual_accuracy, "o-", label="learned policy")
plt.xlabel("stated confidence"); plt.ylabel("actual accuracy"); plt.legend()
plt.title("Calibration of eviction confidence"); plt.tight_layout()
plt.savefig("results/fig_calibration.png", dpi=150)
print("\nSaved results/bonus_explanations.csv, calibration.csv, fig_calibration.png")
