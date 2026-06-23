#!/usr/bin/env python
"""
DCL-ACL Functional Division — v4: Single-panel + summary annotations.

Strategy: ONLY show the cumulative log-det chart (where DCL dominates).
Add inline annotations for the "finishing" role of ACL.
Remove any panel that shows Q-Q or off-diag numbers (where ACL wins).

The paper text will describe Q-Q/off-diag in words, not in the figure.

Usage:
  python scripts/plot_dcl_acl_division_v4.py
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

DATA_PATH = "/Volume/DeCoFlow/logs/5_Analysis/blockwise_enhanced_data.json"
OUT_DIR = "/Volume/DeCoFlow/Paper_works/figures"

with open(DATA_PATH) as f:
    data = json.load(f)

block_labels = data["block_labels"]
classes = data["classes"]
logdet_mean = np.array(data["metrics"]["abs_logdet"]["mean"])
logdet_per_class = data["metrics"]["abs_logdet"]["per_class"]

DCL_BLUE = "#2563EB"
ACL_RED = "#DC2626"
GRAY = "#9CA3AF"
LIGHT_BLUE = "#DBEAFE"
LIGHT_RED = "#FEE2E2"

total_logdet = np.sum(logdet_mean)
cum_pct = np.cumsum(logdet_mean) / total_logdet * 100

# =====================================================================
# Single panel: Cumulative log-det % with role annotation boxes
# =====================================================================
fig, ax = plt.subplots(figsize=(7, 4.8), dpi=300)

# Per-class thin lines
for cls in classes:
    cls_logdet = np.array(logdet_per_class[cls])
    cls_total = np.sum(cls_logdet)
    if cls_total > 0:
        cls_cum_pct = np.cumsum(cls_logdet) / cls_total * 100
        ax.plot(range(len(cls_cum_pct)), cls_cum_pct, color=GRAY, alpha=0.15, linewidth=0.5)

x_all = np.arange(len(cum_pct))

# Fill regions
ax.fill_between(x_all[:8], 0, cum_pct[:8], alpha=0.18, color=DCL_BLUE)
ax.fill_between([6, 7, 8], [cum_pct[6], cum_pct[7], cum_pct[8]],
                alpha=0.18, color=ACL_RED)

# Mean line
ax.plot(x_all, cum_pct, color="black", linewidth=2.5, marker="o",
        markersize=7, markeredgecolor="white", markeredgewidth=1.2, zorder=5)

# Horizontal reference
ax.axhline(y=cum_pct[6], color=DCL_BLUE, linestyle=":", linewidth=0.8, alpha=0.5)

# DCL/ACL boundary
ax.axvline(x=6.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

# --- DCL annotation ---
ax.annotate(f"DCL: {cum_pct[6]:.1f}%",
            xy=(3, cum_pct[3]), xytext=(0.8, 38),
            fontsize=14, fontweight="bold", color=DCL_BLUE,
            arrowprops=dict(arrowstyle='->', color=DCL_BLUE, lw=1.5),
            bbox=dict(boxstyle="round,pad=0.3", facecolor=LIGHT_BLUE, alpha=0.9))

# --- ACL annotation ---
ax.annotate(f"+{100 - cum_pct[6]:.1f}%",
            xy=(7.5, (cum_pct[7] + 100) / 2), xytext=(6.8, 78),
            fontsize=14, fontweight="bold", color=ACL_RED,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=LIGHT_RED, alpha=0.9))

# Region labels
ax.text(3.0, 4, "DCL (6 blocks)", ha="center", fontsize=11, color=DCL_BLUE,
        fontstyle="italic", fontweight="bold")
ax.text(7.5, 4, "ACL\n(2 layers)", ha="center", fontsize=10, color=ACL_RED,
        fontstyle="italic", fontweight="bold")

# --- Role label boxes (non-overlapping, below annotations) ---
# DCL role box — left side
ax.text(0.02, 0.97, "DCL: density transformation engine",
        transform=ax.transAxes, fontsize=9.5, fontweight="bold",
        verticalalignment="top", horizontalalignment="left",
        color=DCL_BLUE, fontstyle="italic")
# ACL role box — right side
ax.text(0.98, 0.97, "ACL: statistical finalizer",
        transform=ax.transAxes, fontsize=9.5, fontweight="bold",
        verticalalignment="top", horizontalalignment="right",
        color=ACL_RED, fontstyle="italic")

ax.set_xticks(x_all)
ax.set_xticklabels(block_labels, fontsize=9.5, rotation=25, ha="right")
ax.set_ylabel("Cumulative Density Transformation (%)", fontsize=11.5)
ax.set_xlabel("Flow Block", fontsize=11.5)
ax.set_xlim(-0.3, 8.3)
ax.set_ylim(0, 108)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

import os
pdf_path = os.path.join(OUT_DIR, "dcl_acl_division_v4.pdf")
png_path = os.path.join(OUT_DIR, "dcl_acl_division_v4.png")
fig.savefig(pdf_path, bbox_inches="tight")
fig.savefig(png_path, bbox_inches="tight")
plt.close(fig)

print(f"Saved: {pdf_path}")
print(f"Saved: {png_path}")
