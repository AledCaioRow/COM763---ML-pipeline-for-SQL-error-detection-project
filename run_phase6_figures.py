# -*- coding: utf-8 -*-
"""Generate figures 13 and 14 for Phase 6 of PROJECT_NARRATIVE_REPORT."""
import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

PROJECT_ROOT = r"C:\Users\aled_\Downloads\COM763---ML-pipeline-for-SQL-error-detection-project"
FIG_DIR      = os.path.join(PROJECT_ROOT, "reports", "narrative_figures")
CSV_PATH     = os.path.join(PROJECT_ROOT, "reports", "within_db_schema_metrics.csv")

df = pd.read_csv(CSV_PATH)

# Best model per database (highest r2_log)
best = (df.sort_values("r2_log", ascending=False)
          .groupby("db_id", sort=False)
          .first()
          .reset_index()
          .sort_values("r2_log", ascending=True))   # ascending for horizontal bar

# -----------------------------------------------------------------------
# Figure 13 -- within-database R2(log) per database, best model
# -----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))

colors = ["#d32f2f" if v < 0 else "#388e3c" for v in best["r2_log"]]
bars = ax.barh(best["db_id"], best["r2_log"], color=colors, edgecolor="white", height=0.6)

ax.axvline(0, color="black", linewidth=1.2, linestyle="--", alpha=0.7)

# Annotate each bar
for bar, val, model in zip(bars, best["r2_log"], best["model"]):
    x = val + (0.02 if val >= 0 else -0.02)
    ha = "left" if val >= 0 else "right"
    ax.text(x, bar.get_y() + bar.get_height() / 2,
            f"R2={val:.3f}  ({model})", va="center", ha=ha, fontsize=8.5)

ax.set_xlabel("R2 (log scale) -- best model per database", fontsize=11)
ax.set_title("Figure 13: Within-Database Regression R2 (log)\n"
             "Each database trained and tested independently (80/20 split)",
             fontsize=12, fontweight="bold")
ax.set_xlim(-1.4, 1.25)

green_patch = mpatches.Patch(color="#388e3c", label="R2 > 0 (beats naive mean)")
red_patch   = mpatches.Patch(color="#d32f2f", label="R2 < 0 (worse than naive mean)")
ax.legend(handles=[green_patch, red_patch], loc="lower right", fontsize=9)

plt.tight_layout()
out = os.path.join(FIG_DIR, "fig13_within_db_r2.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out}")

# -----------------------------------------------------------------------
# Figure 14 -- schema stats vs R2: two scatter panels
# -----------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# Panel A -- index coverage vs R2
ax = axes[0]
for _, row in best.iterrows():
    c = "#388e3c" if row["r2_log"] >= 0 else "#d32f2f"
    ax.scatter(row["index_coverage"], row["r2_log"], color=c, s=90, zorder=3, edgecolors="white")
    ax.annotate(row["db_id"].replace("_", "\n"), (row["index_coverage"], row["r2_log"]),
                textcoords="offset points", xytext=(6, 4), fontsize=7.5)

ax.axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.6)
ax.set_xlabel("Index coverage (fraction of tables indexed)", fontsize=10)
ax.set_ylabel("R2 (log)", fontsize=10)
ax.set_title("Index Coverage vs Model R2", fontsize=11, fontweight="bold")
ax.set_xlim(-0.08, 1.15)

# Panel B -- log10(total_rows) vs R2
ax = axes[1]
for _, row in best.iterrows():
    c = "#388e3c" if row["r2_log"] >= 0 else "#d32f2f"
    ax.scatter(np.log10(row["total_rows"] + 1), row["r2_log"], color=c, s=90, zorder=3, edgecolors="white")
    ax.annotate(row["db_id"].replace("_", "\n"),
                (np.log10(row["total_rows"] + 1), row["r2_log"]),
                textcoords="offset points", xytext=(6, 4), fontsize=7.5)

ax.axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.6)
ax.set_xlabel("log10(total database rows)", fontsize=10)
ax.set_ylabel("R2 (log)", fontsize=10)
ax.set_title("Database Size vs Model R2", fontsize=11, fontweight="bold")

green_patch = mpatches.Patch(color="#388e3c", label="R2 > 0")
red_patch   = mpatches.Patch(color="#d32f2f", label="R2 < 0")
axes[1].legend(handles=[green_patch, red_patch], fontsize=9)

fig.suptitle("Figure 14: Does Schema Complexity Predict Whether the Model Fails?\n"
             "Within-database experiments (80/20 split, best model per DB)",
             fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout()
out = os.path.join(FIG_DIR, "fig14_schema_vs_r2.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out}")

print("Phase 6 figures done.")
