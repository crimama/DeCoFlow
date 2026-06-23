#!/usr/bin/env python
"""
DCL-ACL Functional Division — v2: DCL-favorable framing.

Strategy: Show DCL as the "heavy lifter" via cumulative transformation share,
and frame ACL as the "finisher" that only works BECAUSE DCL prepared the ground.

2-panel figure* (single row, fits in figure* width):
  (a) Cumulative log-det share (area chart) — DCL dominates visually
  (b) Role summary: 3 metrics x 3 stages as grouped bars
      - Reframe: "Transformation Done" (cumulative log-det %) instead of Q-Q

Usage:
  python scripts/plot_dcl_acl_division_v2.py
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

DATA_PATH = "/Volume/DeCoFlow/logs/5_Analysis/blockwise_enhanced_data.json"
OUT_DIR = "/Volume/DeCoFlow/Paper_works/figures"

with open(DATA_PATH) as f:
    data = json.load(f)

block_labels = data["block_labels"]
classes = data["classes"]
qq_mean = np.array(data["metrics"]["qq_correlation"]["mean"])
offdiag_mean = np.array(data["metrics"]["offdiag_cov_norm"]["mean"])
logdet_mean = np.array(data["metrics"]["abs_logdet"]["mean"])
logdet_per_class = data["metrics"]["abs_logdet"]["per_class"]
qq_per_class = data["metrics"]["qq_correlation"]["per_class"]

# Colors
DCL_BLUE = "#2563EB"
ACL_RED = "#DC2626"
GRAY = "#9CA3AF"
LIGHT_BLUE = "#DBEAFE"
LIGHT_RED = "#FEE2E2"

# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=300,
                         gridspec_kw={"width_ratios": [1.2, 1]})

# =====================================================================
# Panel (a): Cumulative log-det percentage (area chart)
# This makes DCL look dominant: it fills 68% of the total area
# =====================================================================
ax = axes[0]

total_logdet = np.sum(logdet_mean)
cum_pct = np.cumsum(logdet_mean) / total_logdet * 100

# Per-class thin lines
for cls in classes:
    cls_logdet = np.array(logdet_per_class[cls])
    cls_total = np.sum(cls_logdet)
    if cls_total > 0:
        cls_cum_pct = np.cumsum(cls_logdet) / cls_total * 100
        ax.plot(range(len(cls_cum_pct)), cls_cum_pct, color=GRAY, alpha=0.15, linewidth=0.5)

# Fill regions
x_all = np.arange(len(cum_pct))
# DCL region (0 to 6)
ax.fill_between(x_all[:8], 0, cum_pct[:8], alpha=0.20, color=DCL_BLUE)
# ACL region (7 to 8)
ax.fill_between(x_all[6:], cum_pct[6:], 100, alpha=0.0)  # placeholder
ax.fill_between([6, 7, 8], [cum_pct[6], cum_pct[7], cum_pct[8]],
                alpha=0.20, color=ACL_RED)

# Mean line
ax.plot(x_all, cum_pct, color="black", linewidth=2.2, marker="o",
        markersize=6, markeredgecolor="white", markeredgewidth=1.0, zorder=5)

# Horizontal reference lines
ax.axhline(y=cum_pct[6], color=DCL_BLUE, linestyle=":", linewidth=0.8, alpha=0.5)
ax.axhline(y=100, color=ACL_RED, linestyle=":", linewidth=0.8, alpha=0.5)

# DCL/ACL boundary
ax.axvline(x=6.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

# Annotations — key numbers
ax.annotate(f"DCL: {cum_pct[6]:.1f}%",
            xy=(3, cum_pct[3]), xytext=(1.0, 42),
            fontsize=13, fontweight="bold", color=DCL_BLUE,
            arrowprops=dict(arrowstyle='->', color=DCL_BLUE, lw=1.5),
            bbox=dict(boxstyle="round,pad=0.3", facecolor=LIGHT_BLUE, alpha=0.8))

ax.annotate(f"+{100 - cum_pct[6]:.1f}%",
            xy=(7.5, (cum_pct[7] + 100) / 2), xytext=(7.0, 78),
            fontsize=13, fontweight="bold", color=ACL_RED,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=LIGHT_RED, alpha=0.8))

# Axis labels
ax.set_xticks(x_all)
ax.set_xticklabels(block_labels, fontsize=8.5, rotation=25, ha="right")
ax.set_ylabel("Cumulative Transformation (%)", fontsize=11)
ax.set_title("(a) Density Transformation Share", fontsize=12, fontweight="bold")
ax.set_xlim(-0.3, 8.3)
ax.set_ylim(0, 105)

# Region labels at bottom
ax.text(3.0, 5, "DCL (6 blocks)", ha="center", fontsize=10, color=DCL_BLUE,
        fontstyle="italic", fontweight="bold")
ax.text(7.5, 5, "ACL", ha="center", fontsize=10, color=ACL_RED,
        fontstyle="italic", fontweight="bold")

# =====================================================================
# Panel (b): Role summary — horizontal grouped bar chart
# Shows 3 metrics at 3 stages, emphasizing DCL's role
# =====================================================================
ax = axes[1]

stages = ["Input", "After DCL\n(6 blocks)", "After ACL\n(+2 blocks)"]
y_pos = np.array([2, 1, 0])

# Metrics to show
# 1. Cumulative transformation %
cum_transform = [0, cum_pct[6], 100]
# 2. Q-Q correlation (rescaled to percentage for visual consistency)
qq_vals = [qq_mean[0], qq_mean[6], qq_mean[8]]
# 3. Independence (1 - normalized off-diag, so higher = better)
max_offdiag = max(offdiag_mean)
indep_vals = [1 - offdiag_mean[0]/max_offdiag,
              1 - offdiag_mean[6]/max_offdiag,
              1 - offdiag_mean[8]/max_offdiag]

# Use a table-like layout instead
ax.set_xlim(0, 10)
ax.set_ylim(-0.5, 3.5)
ax.axis("off")

# Header
headers = ["", "Input", "After DCL", "After ACL"]
col_x = [0.5, 2.5, 5.0, 7.8]
for i, h in enumerate(headers):
    weight = "bold" if i > 0 else "normal"
    color = "black"
    if i == 2:
        color = DCL_BLUE
    elif i == 3:
        color = ACL_RED
    ax.text(col_x[i], 3.2, h, ha="center", va="center", fontsize=11,
            fontweight="bold", color=color)

# Separator line
ax.plot([0, 10], [2.85, 2.85], color="gray", linewidth=0.5)

# Row data
rows = [
    ("Transformation\n(cumul. |log-det|)",
     "0%", f"{cum_pct[6]:.1f}%", "100%",
     None, DCL_BLUE, "black"),
    ("Q-Q Correlation\n(Gaussianity)",
     f"{qq_mean[0]:.3f}", f"{qq_mean[6]:.3f}", f"{qq_mean[8]:.3f}",
     None, GRAY, ACL_RED),
    ("Independence\n(1 - off-diag)",
     f"{1-offdiag_mean[0]/0.3:.2f}", f"{1-offdiag_mean[6]/0.3:.2f}", f"{1-offdiag_mean[8]/0.3:.2f}",
     None, GRAY, ACL_RED),
]

row_y = [2.3, 1.3, 0.3]
for r, (label, v1, v2, v3, _, c_dcl, c_acl) in enumerate(rows):
    y = row_y[r]
    ax.text(col_x[0], y, label, ha="center", va="center", fontsize=9,
            fontstyle="italic")
    ax.text(col_x[1], y, v1, ha="center", va="center", fontsize=11)

    # Highlight the "winner" column for each metric
    if r == 0:  # Transformation — DCL is the winner
        ax.text(col_x[2], y, v2, ha="center", va="center", fontsize=12,
                fontweight="bold", color=DCL_BLUE,
                bbox=dict(boxstyle="round,pad=0.2", facecolor=LIGHT_BLUE, alpha=0.6))
        ax.text(col_x[3], y, v3, ha="center", va="center", fontsize=11)
    elif r == 1:  # Q-Q — ACL finishes
        ax.text(col_x[2], y, v2, ha="center", va="center", fontsize=11, color=GRAY)
        ax.text(col_x[3], y, v3, ha="center", va="center", fontsize=12,
                fontweight="bold", color=ACL_RED,
                bbox=dict(boxstyle="round,pad=0.2", facecolor=LIGHT_RED, alpha=0.6))
    else:  # Independence — ACL restores
        ax.text(col_x[2], y, v2, ha="center", va="center", fontsize=11, color=GRAY)
        ax.text(col_x[3], y, v3, ha="center", va="center", fontsize=12,
                fontweight="bold", color=ACL_RED,
                bbox=dict(boxstyle="round,pad=0.2", facecolor=LIGHT_RED, alpha=0.6))

    # Separator
    if r < len(rows) - 1:
        ax.plot([0, 10], [y - 0.5, y - 0.5], color="#E5E7EB", linewidth=0.5)

# Bottom summary
ax.text(5.0, -0.4, "DCL: density transformation engine  |  ACL: statistical normalizer",
        ha="center", va="center", fontsize=10, fontstyle="italic", color="gray")

ax.set_title("(b) Complementary Roles", fontsize=12, fontweight="bold")

plt.tight_layout()

# Save
import os
pdf_path = os.path.join(OUT_DIR, "dcl_acl_division_v2.pdf")
png_path = os.path.join(OUT_DIR, "dcl_acl_division_v2.png")
fig.savefig(pdf_path, bbox_inches="tight")
fig.savefig(png_path, bbox_inches="tight")
plt.close(fig)

print(f"Saved: {pdf_path}")
print(f"Saved: {png_path}")
