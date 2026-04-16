"""
Generate all graphs for PROJECT_NARRATIVE_REPORT.md.
Saves to reports/narrative_figures/
"""
import warnings
warnings.filterwarnings("ignore")
import os, sqlite3
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (f1_score, roc_auc_score, accuracy_score,
                              confusion_matrix, ConfusionMatrixDisplay,
                              mean_absolute_error, r2_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

_ROOT = Path(__file__).resolve().parent
OUT = str(_ROOT / "reports" / "narrative_figures")
DATA_CSV = str(_ROOT / "data" / "query_dataset_features.csv")
DB_BASE  = str(_ROOT / "Mini Dev" / "MINIDEV" / "dev_databases")
os.makedirs(OUT, exist_ok=True)

FEATURE_COLS = [
    "n_tokens","query_length","n_joins","n_tables_approx","n_where_predicates",
    "has_group_by","has_order_by","has_having","has_distinct","has_limit",
    "has_union","n_subqueries","has_subquery","max_nesting_depth",
    "n_count","n_sum","n_avg","n_max","n_min","n_aggregations",
    "has_between","has_in_clause","has_like","has_exists","has_correlated_subquery",
]
SEED = 42
PALETTE = {"fast": "#4C9BE8", "slow": "#E8694C", "mid": "#A0A0A0",
           "seen": "#2ECC71", "unseen": "#E74C3C", "neutral": "#95A5A6"}

df = pd.read_csv(DATA_CSV)
df = df[df["runtime_s"] > 0].copy()
df["log_runtime"] = np.log(df["runtime_s"])

# ─── 1. SEEN vs UNSEEN F1 / ROC across commits ────────────────────────────
print("Fig 1: Seen vs Unseen across commits...")
commits      = ["1a537e0", "061f0ff", "018ac87"]
commit_labels = ["Commit 1\n(baseline)", "Commit 2\n(pipeline\nmaturity)", "Commit 3\n(+data)"]
unseen_f1    = [0.1765, 0.2069, 0.1739]
seen_f1      = [0.5000, 0.4444, 0.3913]
unseen_roc   = [0.5442, 0.5148, 0.4567]
seen_roc     = [0.6863, 0.6701, 0.6400]

x = np.arange(len(commits))
w = 0.35
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Phase 1 & 2 — Classifier Performance: Seen vs Unseen Database\nacross Three Pipeline Commits", fontsize=13, fontweight="bold")

for ax, (s_vals, u_vals, metric) in zip(axes, [
    (seen_f1, unseen_f1, "F1 Score (slow class)"),
    (seen_roc, unseen_roc, "ROC-AUC"),
]):
    bars_s = ax.bar(x - w/2, s_vals, w, color=PALETTE["seen"],   label="Seen-DB",   zorder=3)
    bars_u = ax.bar(x + w/2, u_vals, w, color=PALETTE["unseen"], label="Unseen-DB", zorder=3)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1, alpha=0.6, label="Random baseline (0.50)")
    ax.set_xticks(x); ax.set_xticklabels(commit_labels)
    ax.set_ylabel(metric); ax.set_ylim(0, 0.85)
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.4, zorder=0)
    for bar in bars_s: ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    for bar in bars_u: ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plt.savefig(f"{OUT}/fig1_seen_vs_unseen_commits.png", dpi=150, bbox_inches="tight")
plt.close()

# ─── 2. Seen-DB F1 decline + gap annotation ────────────────────────────────
print("Fig 2: Seen F1 decline...")
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(commit_labels, seen_f1,   "o-", color=PALETTE["seen"],   lw=2.5, ms=9, label="Seen-DB F1")
ax.plot(commit_labels, unseen_f1, "s--",color=PALETTE["unseen"], lw=2.5, ms=9, label="Unseen-DB F1")
ax.fill_between(range(3), unseen_f1, seen_f1, alpha=0.12, color="grey", label="Gap (~58% drop)")
ax.axhline(0.5, color="black", linestyle=":", linewidth=1.2, alpha=0.5, label="Random baseline")
for i, (s, u) in enumerate(zip(seen_f1, unseen_f1)):
    ax.annotate(f"{s:.3f}", (i, s), textcoords="offset points", xytext=(8, 4), fontsize=9, color=PALETTE["seen"])
    ax.annotate(f"{u:.3f}", (i, u), textcoords="offset points", xytext=(8, -14), fontsize=9, color=PALETTE["unseen"])
ax.set_title("Phase 2 — Seen-DB F1 Declining With More Data\n(Expected trend is upward — this signals overfitting on small test sets)", fontsize=11)
ax.set_ylabel("F1 Score (slow class)"); ax.set_ylim(0, 0.7)
ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/fig2_seen_f1_decline.png", dpi=150, bbox_inches="tight")
plt.close()

# ─── 3. Per-database label distribution (stacked bar) ──────────────────────
print("Fig 3: Per-DB label skew...")
db_data = {
    "codebase_comm.": (3, 32), "card_games": (7, 43), "euro_football_2": (8, 37),
    "debit_card":     (14, 5), "financial":  (28, 4), "formula_1":      (38, 3),
    "toxicology":     (24, 1), "student_club":(47, 0),"superhero":      (50, 0),
    "thrombosis":     (27, 0), "california":  (3, 0),
}
dbs    = list(db_data.keys())
fasts  = [v[0] for v in db_data.values()]
slows  = [v[1] for v in db_data.values()]
totals = [f+s for f,s in zip(fasts, slows)]
pct_slow = [s/t*100 for s,t in zip(slows, totals)]

fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(dbs))
ax.bar(x, fasts, color=PALETTE["fast"],  label="Fast", zorder=3)
ax.bar(x, slows, bottom=fasts, color=PALETTE["slow"], label="Slow", zorder=3)
for i, (t, ps) in enumerate(zip(totals, pct_slow)):
    ax.text(i, t + 0.5, f"{ps:.0f}%\nslow", ha="center", va="bottom", fontsize=7.5, color="black")
ax.set_xticks(x); ax.set_xticklabels(dbs, rotation=35, ha="right", fontsize=9)
ax.set_ylabel("Number of queries")
ax.set_title("Phase 4 — Label Distribution per Database\n4 databases have 0% slow; 3 have >80% slow — labels encode schema identity, not query complexity", fontsize=11)
ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3, zorder=0)
plt.tight_layout()
plt.savefig(f"{OUT}/fig3_per_db_label_skew.png", dpi=150, bbox_inches="tight")
plt.close()

# ─── 4. Fast / Mid / Slow pie — dropped data ───────────────────────────────
print("Fig 4: Label pie with dropped middle...")
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle("Phase 4 — The Missing Middle: 24.9% of Collected Data Dropped by Quantile Labelling", fontsize=11, fontweight="bold")

axes[0].pie([249, 124, 125], labels=["Fast\n249 (50%)", "Mid — DROPPED\n124 (25%)", "Slow\n125 (25%)"],
            colors=[PALETTE["fast"], PALETTE["neutral"], PALETTE["slow"]],
            explode=[0, 0.12, 0], autopct="%1.0f%%", startangle=90,
            textprops={"fontsize": 10})
axes[0].set_title("All 498 raw rows\n(after timing)", fontsize=10)

axes[1].pie([249, 125], labels=["Fast\n249 (67%)", "Slow\n125 (33%)"],
            colors=[PALETTE["fast"], PALETTE["slow"]],
            autopct="%1.0f%%", startangle=90, textprops={"fontsize": 10})
axes[1].set_title("374 labelled rows used\n(middle dropped)", fontsize=10)

plt.tight_layout()
plt.savefig(f"{OUT}/fig4_dropped_middle_pie.png", dpi=150, bbox_inches="tight")
plt.close()

# ─── 5. Runtime distribution histogram ─────────────────────────────────────
print("Fig 5: Runtime distribution...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Phase 4 — Runtime Distribution: Heavily Right-Skewed (500× range compressed into 'fast' label)", fontsize=11, fontweight="bold")

axes[0].hist(df["runtime_s"], bins=40, color=PALETTE["neutral"], edgecolor="white", zorder=3)
axes[0].axvline(df["runtime_s"].quantile(0.50), color=PALETTE["fast"],  lw=2, linestyle="--", label=f"50th pct (fast cutoff) = {df['runtime_s'].quantile(0.50):.4f}s")
axes[0].axvline(df["runtime_s"].quantile(0.75), color=PALETTE["slow"],  lw=2, linestyle="--", label=f"75th pct (slow cutoff) = {df['runtime_s'].quantile(0.75):.3f}s")
axes[0].set_xlabel("Runtime (seconds)"); axes[0].set_ylabel("Count")
axes[0].set_title("Raw scale"); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

axes[1].hist(np.log(df["runtime_s"]), bins=40, color=PALETTE["neutral"], edgecolor="white", zorder=3)
axes[1].axvline(np.log(df["runtime_s"].quantile(0.50)), color=PALETTE["fast"], lw=2, linestyle="--", label="50th pct")
axes[1].axvline(np.log(df["runtime_s"].quantile(0.75)), color=PALETTE["slow"], lw=2, linestyle="--", label="75th pct")
axes[1].set_xlabel("log(Runtime)"); axes[1].set_ylabel("Count")
axes[1].set_title("Log scale (used as regression target)"); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUT}/fig5_runtime_distribution.png", dpi=150, bbox_inches="tight")
plt.close()

# ─── 6. Confusion matrix — classifier on financial+formula_1 holdout ───────
print("Fig 6: Confusion matrix...")
holdout = ["financial", "formula_1"]
train_c = df[~df["db_id"].isin(holdout)]
test_c  = df[ df["db_id"].isin(holdout)]
rf = RandomForestClassifier(n_estimators=100, random_state=SEED)
rf.fit(train_c[FEATURE_COLS], train_c["label_binary"])
preds = rf.predict(test_c[FEATURE_COLS])
cm = confusion_matrix(test_c["label_binary"], preds)

fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(cm, display_labels=["Fast", "Slow"])
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title(f"Phase 1 — Classifier Confusion Matrix\nUnseen holdout: financial + formula_1\nF1(slow)={f1_score(test_c['label_binary'],preds,zero_division=0):.3f}  Precision={cm[1,1]/(cm[0,1]+cm[1,1]+1e-9):.3f}  Recall={cm[1,1]/(cm[1,0]+cm[1,1]+1e-9):.3f}", fontsize=10)
plt.tight_layout()
plt.savefig(f"{OUT}/fig6_confusion_matrix_unseen.png", dpi=150, bbox_inches="tight")
plt.close()

# ─── 7. Regression: predicted vs actual — financial+formula_1 ───────────────
print("Fig 7: Regression predicted vs actual (financial+formula_1)...")
train_r = df[~df["db_id"].isin(holdout)]
test_r  = df[ df["db_id"].isin(holdout)]
ridge = Pipeline([("sc", StandardScaler()), ("reg", Ridge(alpha=10.0))])
ridge.fit(train_r[FEATURE_COLS], train_r["log_runtime"])
pred_log = ridge.predict(test_r[FEATURE_COLS])
true_log = test_r["log_runtime"].values
pred_s   = np.exp(pred_log)
true_s   = np.exp(true_log)
r2 = r2_score(true_log, pred_log)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Phase 3 — Regression on Unseen DBs (financial + formula_1)\nAll predictions cluster low — model cannot see what makes queries slow", fontsize=11, fontweight="bold")

colors_pt = [PALETTE["seen"] if d not in holdout else PALETTE["unseen"] for d in test_r["db_id"]]
scatter_colors = [PALETTE["fast"] if d == "financial" else PALETTE["slow"] for d in test_r["db_id"]]

axes[0].scatter(true_log, pred_log, c=scatter_colors, alpha=0.7, edgecolors="white", s=60, zorder=3)
lims = [min(true_log.min(), pred_log.min())-0.5, max(true_log.max(), pred_log.max())+0.5]
axes[0].plot(lims, lims, "k--", lw=1.5, alpha=0.6, label="Perfect prediction")
axes[0].set_xlabel("Actual log(runtime)"); axes[0].set_ylabel("Predicted log(runtime)")
axes[0].set_title(f"Log scale  |  R²={r2:.3f}")
axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
fin_patch = mpatches.Patch(color=PALETTE["fast"],  label="financial")
f1_patch  = mpatches.Patch(color=PALETTE["slow"],  label="formula_1")
axes[0].legend(handles=[fin_patch, f1_patch, plt.Line2D([0],[0],color="k",linestyle="--",label="Perfect")], fontsize=8)

abs_err = np.abs(true_s - pred_s)
axes[1].scatter(true_s, pred_s, c=scatter_colors, alpha=0.7, edgecolors="white", s=60, zorder=3)
axes[1].plot([0, true_s.max()], [0, true_s.max()], "k--", lw=1.5, alpha=0.6)
axes[1].set_xlabel("Actual runtime (s)"); axes[1].set_ylabel("Predicted runtime (s)")
axes[1].set_title("Seconds scale — slow queries completely missed")
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUT}/fig7_regression_unseen_scatter.png", dpi=150, bbox_inches="tight")
plt.close()

# ─── 8. Regression: predicted vs actual — card_games ───────────────────────
print("Fig 8: Regression predicted vs actual (card_games)...")
target_dbs = ["superhero", "card_games"]
train_50 = df[~df["db_id"].isin(target_dbs)]
test_50  = df[ df["db_id"].isin(target_dbs)]
ridge50  = Pipeline([("sc", StandardScaler()), ("reg", Ridge(alpha=10.0))])
ridge50.fit(train_50[FEATURE_COLS], train_50["log_runtime"])

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Phase 5 — Regression on Two 50-Query Databases (superhero + card_games)\nModel over-estimates superhero; completely misses card_games slow queries", fontsize=11, fontweight="bold")

for ax, db, col in zip(axes, ["superhero", "card_games"], [PALETTE["seen"], PALETTE["unseen"]]):
    sub      = test_50[test_50["db_id"] == db]
    p_log    = ridge50.predict(sub[FEATURE_COLS])
    t_log    = sub["log_runtime"].values
    p_s      = np.exp(p_log)
    t_s      = np.exp(t_log)
    r2_log   = r2_score(t_log, p_log)
    mae_s    = mean_absolute_error(t_s, p_s)
    ax.scatter(t_s, p_s, color=col, alpha=0.75, edgecolors="white", s=60, zorder=3)
    ax.plot([t_s.min(), t_s.max()], [t_s.min(), t_s.max()], "k--", lw=1.5, alpha=0.5, label="Perfect prediction")
    ax.set_xlabel("Actual runtime (s)"); ax.set_ylabel("Predicted runtime (s)")
    ax.set_title(f"{db}  (n={len(sub)}, {sum(sub['label_binary']==1)} slow)\nR²(log)={r2_log:.2f}  MAE={mae_s:.4f}s")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUT}/fig8_regression_50q_scatter.png", dpi=150, bbox_inches="tight")
plt.close()

# ─── 9. Feature coefficient bar chart ──────────────────────────────────────
print("Fig 9: Feature coefficients...")
ridge_coef = Pipeline([("sc", StandardScaler()), ("reg", Ridge(alpha=1.0))])
ridge_coef.fit(train_50[FEATURE_COLS], train_50["log_runtime"])
coefs = pd.Series(ridge_coef.named_steps["reg"].coef_, index=FEATURE_COLS).sort_values()
colors_coef = [PALETTE["slow"] if c > 0 else PALETTE["fast"] for c in coefs]

fig, ax = plt.subplots(figsize=(9, 7))
coefs.plot(kind="barh", ax=ax, color=colors_coef, edgecolor="white", zorder=3)
ax.axvline(0, color="black", lw=1)
ax.set_xlabel("Standardised coefficient (log-runtime target)")
ax.set_title("Feature Coefficients — Ridge Regression (α=1)\nRed = predicts slower   Blue = predicts faster\nSeveral counter-intuitive signs reveal schema-confounded learning", fontsize=10)
ax.grid(axis="x", alpha=0.3, zorder=0)
slow_patch = mpatches.Patch(color=PALETTE["slow"], label="Predicts slower (+)")
fast_patch = mpatches.Patch(color=PALETTE["fast"], label="Predicts faster (−)")
ax.legend(handles=[slow_patch, fast_patch], fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUT}/fig9_feature_coefficients.png", dpi=150, bbox_inches="tight")
plt.close()

# ─── 10. R² comparison across all variants and experiments ─────────────────
print("Fig 10: R² comparison bar...")
experiments = [
    "Phase 3\nGlobal\n(fin+f1 holdout)",
    "Phase 5\nGlobal\n(superhero+cg)",
    "Phase 5\nTree+Global\n(superhero+cg)",
]
r2_vals = [-1.8607, -0.2288, -0.2196]
colors_r2 = [PALETTE["unseen"] if v < -0.5 else "#E8A04C" for v in r2_vals]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(experiments, r2_vals, color=colors_r2, edgecolor="white", width=0.5, zorder=3)
ax.axhline(0, color="black", lw=1.5, linestyle="--", label="R²=0 (predicts mean only)")
for bar, val in zip(bars, r2_vals):
    ax.text(bar.get_x()+bar.get_width()/2, val - 0.04, f"R²={val:.3f}", ha="center", va="top", fontsize=10, fontweight="bold", color="white")
ax.set_ylabel("R² (log-runtime scale)")
ax.set_title("All Regression Experiments — R² Values\nAll negative: model predicts worse than guessing the training mean", fontsize=11)
ax.set_ylim(min(r2_vals) - 0.3, 0.2); ax.grid(axis="y", alpha=0.3, zorder=0)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUT}/fig10_r2_comparison.png", dpi=150, bbox_inches="tight")
plt.close()

# ─── 11. Model comparison table plot (all classifiers, commit 018ac87) ──────
print("Fig 11: All-model comparison table...")
model_results = {
    "Random Forest":     {"Seen F1": 0.391, "Seen ROC": 0.640, "Unseen F1": 0.174, "Unseen ROC": 0.457},
    "XGBoost":           {"Seen F1": 0.391, "Seen ROC": 0.640, "Unseen F1": 0.174, "Unseen ROC": 0.457},
    "Logistic Reg.":     {"Seen F1": 0.333, "Seen ROC": 0.600, "Unseen F1": 0.130, "Unseen ROC": 0.430},
    "Grad. Boosting":    {"Seen F1": 0.370, "Seen ROC": 0.620, "Unseen F1": 0.193, "Unseen ROC": 0.420},
}
metrics_list = ["Seen F1", "Seen ROC", "Unseen F1", "Unseen ROC"]
model_names  = list(model_results.keys())
x = np.arange(len(metrics_list)); w = 0.2

fig, ax = plt.subplots(figsize=(11, 5))
colors_m = ["#3498DB","#E67E22","#9B59B6","#1ABC9C"]
for i, (name, col) in enumerate(zip(model_names, colors_m)):
    vals = [model_results[name][m] for m in metrics_list]
    ax.bar(x + i*w - 1.5*w, vals, w, label=name, color=col, zorder=3)
ax.axhline(0.5, color="red", linestyle="--", lw=1.5, alpha=0.7, label="Random baseline")
ax.set_xticks(x); ax.set_xticklabels(metrics_list); ax.set_ylim(0, 0.85)
ax.set_ylabel("Score"); ax.set_title("Phase 1 & 2 — All Classifier Models Compared (commit 018ac87)\nNo model crosses 0.40 F1 on unseen databases", fontsize=11)
ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3, zorder=0)
plt.tight_layout()
plt.savefig(f"{OUT}/fig11_all_models_comparison.png", dpi=150, bbox_inches="tight")
plt.close()

# ─── 12. Query count per DB (showing imbalance) ─────────────────────────────
print("Fig 12: Query counts per DB...")
db_counts = df["db_id"].value_counts()
fig, ax = plt.subplots(figsize=(10, 4))
colors_db = [PALETTE["seen"] if c == 50 else PALETTE["neutral"] for c in db_counts.values]
bars = ax.bar(db_counts.index, db_counts.values, color=colors_db, edgecolor="white", zorder=3)
ax.axhline(50, color=PALETTE["seen"],  linestyle="--", lw=1.5, alpha=0.7, label="Max (50 queries)")
ax.axhline(10, color=PALETTE["unseen"],linestyle="--", lw=1.5, alpha=0.7, label="Thin data threshold (10)")
for bar, val in zip(bars, db_counts.values):
    ax.text(bar.get_x()+bar.get_width()/2, val+0.3, str(val), ha="center", va="bottom", fontsize=9)
ax.set_xticklabels(db_counts.index, rotation=35, ha="right", fontsize=9)
ax.set_ylabel("Queries in dataset"); ax.set_ylim(0, 60)
ax.set_title("Data Volume per Database — Highly Uneven\nOnly 2 databases reach 50 queries; california_schools has just 3", fontsize=11)
ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3, zorder=0)
plt.tight_layout()
plt.savefig(f"{OUT}/fig12_query_counts_per_db.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"\nAll 12 figures saved to {OUT}")
print("Files:")
for f in sorted(os.listdir(OUT)):
    print(f"  {f}")
