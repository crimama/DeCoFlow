#!/usr/bin/env python
"""
Revised Block-wise Analysis Figure (v2)

Three panels — all from pre-computed JSON data (no GPU needed):
  (a) Cumulative Density Transformation (%) — dcl_acl_division_v4 style
  (b) Cross-Dim. Independence (Off-Diagonal Cov Norm) — unchanged
  (c) Marginal Gaussianity (Q-Q Correlation) — moved from old panel (a)

Data source: logs/5_Analysis/blockwise_enhanced_data.json
Output:      Paper_works/figures/blockwise_enhanced_analysis_v2.{pdf,png,jpg}
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os

PROJECT_ROOT = "/Volume/DeCoFlow"
DATA_PATH = os.path.join(PROJECT_ROOT, "logs/5_Analysis/blockwise_enhanced_data.json")
FIG_DIR = os.path.join(PROJECT_ROOT, "Paper_works/figures")

with open(DATA_PATH) as f:
    data = json.load(f)

BLOCK_LABELS = data["block_labels"]  # Input, DCL1..DCL6, ACL1, ACL2
CLASSES = data["classes"]
metrics = data["metrics"]

mean_qq = np.array(metrics["qq_correlation"]["mean"])
std_qq = np.array(metrics["qq_correlation"]["std"])
mean_offdiag = np.array(metrics["offdiag_cov_norm"]["mean"])
std_offdiag = np.array(metrics["offdiag_cov_norm"]["std"])
mean_logdet = np.array(metrics["abs_logdet"]["mean"])
std_logdet = np.array(metrics["abs_logdet"]["std"])

qq_per_class = metrics["qq_correlation"]["per_class"]
offdiag_per_class = metrics["offdiag_cov_norm"]["per_class"]
logdet_per_class = metrics["abs_logdet"]["per_class"]

# ─── Derived: cumulative density transformation ────────────────────
# Cumulative |log det J| as percentage of total
cumul_mean = np.cumsum(mean_logdet)
total_logdet = cumul_mean[-1]
cumul_pct_mean = cumul_mean / total_logdet * 100

# Per-class cumulative %
cumul_pct_per_class = {}
for c in CLASSES:
    ld = np.array(logdet_per_class[c])
    cumul = np.cumsum(ld)
    total = cumul[-1]
    cumul_pct_per_class[c] = (cumul / total * 100).tolist()

# ─── Plot ──────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(17, 5), dpi=300)
x = np.arange(len(BLOCK_LABELS))
boundary_x = 6.5  # between DCL6 and ACL1

BLUE = "#2563EB"
RED = "#DC2626"

# ════════════════════════════════════════════════════════════════════
# Panel (a): Cumulative Density Transformation (%)
# ════════════════════════════════════════════════════════════════════
ax = axes[0]

# Per-class lines (gray)
for c in CLASSES:
    ax.plot(x, cumul_pct_per_class[c],
            color="gray", alpha=0.15, linewidth=0.7, zorder=1)

# Mean line
ax.plot(x, cumul_pct_mean,
        color="black", linewidth=2.5, marker="o", markersize=5,
        markeredgecolor="white", markeredgewidth=0.8, zorder=4)

# Fill areas: DCL (blue), ACL (red)
# DCL region: x=0..6 (Input to DCL6)
ax.fill_between(x[:7], 0, cumul_pct_mean[:7],
                alpha=0.15, color=BLUE, zorder=2)
# ACL region: x=6..8 (DCL6 to ACL2)
ax.fill_between(x[6:], 0, cumul_pct_mean[6:],
                alpha=0.15, color=RED, zorder=2)

# Horizontal reference: DCL contribution boundary
dcl_pct = cumul_pct_mean[6]  # after DCL6
ax.axhline(y=dcl_pct, color=BLUE, linestyle=":", linewidth=0.9, alpha=0.5)

# Vertical boundary
ax.axvline(x=boundary_x, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

# Annotations
ax.annotate(f"DCL: {dcl_pct:.1f}%",
            xy=(3.0, dcl_pct / 2), fontsize=12, color=BLUE,
            fontweight="bold", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=BLUE, alpha=0.85, linewidth=0.8))

acl_pct = 100 - dcl_pct
ax.annotate(f"+{acl_pct:.1f}%",
            xy=(7.5, dcl_pct + acl_pct / 2), fontsize=12, color=RED,
            fontweight="bold", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=RED, alpha=0.85, linewidth=0.8))

# Top labels
ax.text(3.0, 103, "DCL: density transformation engine",
        ha="center", fontsize=10, color=BLUE, fontweight="bold",
        fontstyle="italic")
ax.text(7.5, 103, "ACL: statistical\nfinalizer",
        ha="center", fontsize=10, color=RED, fontweight="bold",
        fontstyle="italic", linespacing=0.9)

ax.set_xticks(x)
ax.set_xticklabels(BLOCK_LABELS, fontsize=10, rotation=30, ha="right")
ax.set_ylabel("Cumulative Density\nTransformation (%)", fontsize=12)
ax.set_title("(a) Cumulative |log det J|", fontsize=14, fontweight="bold")
ax.set_ylim(-2, 112)
ax.set_xlim(-0.3, 8.3)

# ════════════════════════════════════════════════════════════════════
# Panel (b): Off-Diagonal Covariance Norm — unchanged
# ════════════════════════════════════════════════════════════════════
ax = axes[1]
for c in CLASSES:
    ax.plot(x, offdiag_per_class[c],
            color="gray", alpha=0.2, linewidth=0.7, zorder=1)
ax.plot(x, mean_offdiag,
        color=RED, linewidth=2.2, marker="s", markersize=5,
        markeredgecolor="white", markeredgewidth=0.8,
        label="Mean (15 classes)", zorder=3)
ax.fill_between(x, mean_offdiag - std_offdiag, mean_offdiag + std_offdiag,
                alpha=0.12, color=RED, zorder=2)
ax.axvline(x=boundary_x, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
ax.set_xticks(x)
ax.set_xticklabels(BLOCK_LABELS, fontsize=10, rotation=30, ha="right")
ax.set_ylabel(r"Off-Diag. Cov. Norm ($\times D^{-1}$)", fontsize=12)
ax.set_title("(b) Cross-Dim. Independence", fontsize=14, fontweight="bold")
ax.legend(fontsize=10, loc="upper right", framealpha=0.9)
ymin, ymax = ax.get_ylim()
label_y = ymin + 0.03 * (ymax - ymin)
ax.text(3.0, label_y, "DCL", ha="center", fontsize=10, color="gray", style="italic")
ax.text(7.5, label_y, "ACL", ha="center", fontsize=10, color="gray", style="italic")

# ════════════════════════════════════════════════════════════════════
# Panel (c): Q-Q Correlation (Marginal Gaussianity) — was old (a)
# ════════════════════════════════════════════════════════════════════
ax = axes[2]
for c in CLASSES:
    ax.plot(x, qq_per_class[c],
            color="gray", alpha=0.2, linewidth=0.7, zorder=1)
ax.plot(x, mean_qq,
        color=BLUE, linewidth=2.2, marker="o", markersize=5,
        markeredgecolor="white", markeredgewidth=0.8,
        label="Mean (15 classes)", zorder=3)
ax.fill_between(x, mean_qq - std_qq, mean_qq + std_qq,
                alpha=0.12, color=BLUE, zorder=2)
ax.axvline(x=boundary_x, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
ax.set_xticks(x)
ax.set_xticklabels(BLOCK_LABELS, fontsize=10, rotation=30, ha="right")
ax.set_ylabel(r"Q-Q Corr. with $\mathcal{N}(0,1)$", fontsize=12)
ax.set_title("(c) Marginal Gaussianity", fontsize=14, fontweight="bold")
ax.set_ylim(top=1.005)
ax.legend(fontsize=10, loc="upper left", framealpha=0.9)
ymin, ymax = ax.get_ylim()
label_y = ymin + 0.03 * (ymax - ymin)
ax.text(3.0, label_y, "DCL", ha="center", fontsize=10, color="gray", style="italic")
ax.text(7.5, label_y, "ACL", ha="center", fontsize=10, color="gray", style="italic")

# ─── Save ──────────────────────────────────────────────────────────
plt.tight_layout(w_pad=2.0)

for ext in ["pdf", "png", "jpg"]:
    path = os.path.join(FIG_DIR, f"blockwise_enhanced_analysis_v2.{ext}")
    kwargs = {"bbox_inches": "tight"}
    if ext == "jpg":
        kwargs["pil_kwargs"] = {"quality": 95}
    fig.savefig(path, **kwargs)
    print(f"Saved: {path}")

plt.close(fig)

# ─── Summary ───────────────────────────────────────────────────────
print()
print("Panel (a) — Cumulative Density Transformation:")
print(f"  DCL (6 blocks): {dcl_pct:.1f}%")
print(f"  ACL (2 blocks): {acl_pct:.1f}%")
print()
print("Panel (b) — Off-Diagonal Covariance Norm:")
print(f"  Input: {mean_offdiag[0]:.4f} → DCL6: {mean_offdiag[6]:.4f} → ACL2: {mean_offdiag[8]:.4f}")
print()
print("Panel (c) — Q-Q Correlation:")
print(f"  Input: {mean_qq[0]:.4f} → DCL6: {mean_qq[6]:.4f} → ACL2: {mean_qq[8]:.4f}")
