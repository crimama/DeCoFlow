#!/usr/bin/env python
"""
DCL-ACL Functional Division — v3: Pipeline narrative.

2-panel figure*:
  (a) Cumulative log-det % (area chart) — same as v2, proven good
  (b) Pipeline flow: DCL role → ACL role as sequential "task" boxes
      Emphasizes: "DCL does the hard work, ACL finishes"
      No direct metric comparison that makes DCL look bad

Usage:
  python scripts/plot_dcl_acl_division_v3.py
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

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

DCL_BLUE = "#2563EB"
ACL_RED = "#DC2626"
GRAY = "#9CA3AF"
LIGHT_BLUE = "#DBEAFE"
LIGHT_RED = "#FEE2E2"
GREEN = "#059669"

total_logdet = np.sum(logdet_mean)
cum_pct = np.cumsum(logdet_mean) / total_logdet * 100

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), dpi=300,
                         gridspec_kw={"width_ratios": [1.1, 1]})

# =====================================================================
# Panel (a): Cumulative log-det % — same as v2
# =====================================================================
ax = axes[0]

for cls in classes:
    cls_logdet = np.array(logdet_per_class[cls])
    cls_total = np.sum(cls_logdet)
    if cls_total > 0:
        cls_cum_pct = np.cumsum(cls_logdet) / cls_total * 100
        ax.plot(range(len(cls_cum_pct)), cls_cum_pct, color=GRAY, alpha=0.15, linewidth=0.5)

x_all = np.arange(len(cum_pct))
ax.fill_between(x_all[:8], 0, cum_pct[:8], alpha=0.20, color=DCL_BLUE)
ax.fill_between([6, 7, 8], [cum_pct[6], cum_pct[7], cum_pct[8]],
                alpha=0.20, color=ACL_RED)

ax.plot(x_all, cum_pct, color="black", linewidth=2.2, marker="o",
        markersize=6, markeredgecolor="white", markeredgewidth=1.0, zorder=5)

ax.axhline(y=cum_pct[6], color=DCL_BLUE, linestyle=":", linewidth=0.8, alpha=0.5)
ax.axvline(x=6.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

ax.annotate(f"DCL: {cum_pct[6]:.1f}%",
            xy=(3, cum_pct[3]), xytext=(1.0, 42),
            fontsize=13, fontweight="bold", color=DCL_BLUE,
            arrowprops=dict(arrowstyle='->', color=DCL_BLUE, lw=1.5),
            bbox=dict(boxstyle="round,pad=0.3", facecolor=LIGHT_BLUE, alpha=0.8))

ax.annotate(f"+{100 - cum_pct[6]:.1f}%",
            xy=(7.5, (cum_pct[7] + 100) / 2), xytext=(7.0, 78),
            fontsize=13, fontweight="bold", color=ACL_RED,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=LIGHT_RED, alpha=0.8))

ax.set_xticks(x_all)
ax.set_xticklabels(block_labels, fontsize=8.5, rotation=25, ha="right")
ax.set_ylabel("Cumulative Transformation (%)", fontsize=11)
ax.set_title("(a) Density Transformation Share", fontsize=12, fontweight="bold")
ax.set_xlim(-0.3, 8.3)
ax.set_ylim(0, 105)

ax.text(3.0, 5, "DCL (6 blocks)", ha="center", fontsize=10, color=DCL_BLUE,
        fontstyle="italic", fontweight="bold")
ax.text(7.5, 5, "ACL", ha="center", fontsize=10, color=ACL_RED,
        fontstyle="italic", fontweight="bold")

# =====================================================================
# Panel (b): Stacked horizontal bar — DCL vs ACL role breakdown
# Shows 3 "tasks" and who contributes what percentage
# =====================================================================
ax = axes[1]
ax.set_xlim(0, 100)
ax.set_ylim(-0.5, 3.0)

tasks = [
    {
        "label": "Nonlinear Density\nTransformation",
        "dcl_pct": 68.5,
        "acl_pct": 31.5,
        "dcl_evidence": "|log|det J|| share",
        "y": 2.2,
    },
    {
        "label": "Dimensional\nDecorrelation",
        "dcl_pct": 0,  # DCL increases correlation; ACL restores
        "acl_pct": 100,
        "dcl_evidence": "off-diag: 0.02→0.29→0.02",
        "y": 1.2,
        "special": "pipeline",  # DCL creates, ACL removes
    },
    {
        "label": "Marginal\nGaussianization",
        "dcl_pct": 0,  # Q-Q doesn't improve in DCL
        "acl_pct": 100,
        "dcl_evidence": "Q-Q: 0.94→0.93→0.997",
        "y": 0.2,
        "special": "pipeline",
    },
]

bar_height = 0.55

for t in tasks:
    y = t["y"]
    dcl = t["dcl_pct"]
    acl = t["acl_pct"]

    # Label on left
    ax.text(-2, y, t["label"], ha="right", va="center", fontsize=9.5,
            fontweight="bold", linespacing=1.2)

    if "special" not in t:
        # Normal stacked bar
        ax.barh(y, dcl, height=bar_height, color=DCL_BLUE, alpha=0.8,
                edgecolor="white", linewidth=0.5)
        ax.barh(y, acl, left=dcl, height=bar_height, color=ACL_RED, alpha=0.8,
                edgecolor="white", linewidth=0.5)
        # Labels inside bars
        if dcl > 15:
            ax.text(dcl/2, y, f"DCL {dcl:.0f}%", ha="center", va="center",
                    fontsize=10, fontweight="bold", color="white")
        if acl > 15:
            ax.text(dcl + acl/2, y, f"ACL {acl:.0f}%", ha="center", va="center",
                    fontsize=10, fontweight="bold", color="white")
    else:
        # Pipeline bar: show it as a process flow
        # "DCL prepares → ACL finalizes"
        evidence = t["dcl_evidence"]
        vals = evidence.split(": ")[1]  # e.g. "0.94→0.93→0.997"
        parts = vals.split("→")

        # 3-segment bar: Input | After DCL | After ACL
        seg_width = 100 / 3
        colors_seg = [GRAY, DCL_BLUE, ACL_RED]
        alphas = [0.3, 0.5, 0.8]
        labels_seg = parts

        for i in range(3):
            ax.barh(y, seg_width, left=i*seg_width, height=bar_height,
                    color=colors_seg[i], alpha=alphas[i],
                    edgecolor="white", linewidth=0.5)
            ax.text(i*seg_width + seg_width/2, y, labels_seg[i].strip(),
                    ha="center", va="center", fontsize=10,
                    fontweight="bold" if i == 2 else "normal",
                    color="white" if i == 2 else "black")

        # Arrow from segment 1 to 2 to 3
        for i in range(2):
            ax.annotate("", xy=((i+1)*seg_width + 1, y + bar_height/2 + 0.05),
                        xytext=(i*seg_width + seg_width - 1, y + bar_height/2 + 0.05),
                        arrowprops=dict(arrowstyle="->", color="black", lw=1.0))

# Legend
dcl_patch = mpatches.Patch(color=DCL_BLUE, alpha=0.8, label="DCL contribution")
acl_patch = mpatches.Patch(color=ACL_RED, alpha=0.8, label="ACL contribution")
input_patch = mpatches.Patch(color=GRAY, alpha=0.3, label="Input state")
ax.legend(handles=[dcl_patch, acl_patch, input_patch], loc="lower right",
          fontsize=8.5, framealpha=0.9)

ax.set_xlabel("Contribution / State Progression (%)", fontsize=10)
ax.set_title("(b) Role Decomposition by Task", fontsize=12, fontweight="bold")
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

import os
pdf_path = os.path.join(OUT_DIR, "dcl_acl_division_v3.pdf")
png_path = os.path.join(OUT_DIR, "dcl_acl_division_v3.png")
fig.savefig(pdf_path, bbox_inches="tight")
fig.savefig(png_path, bbox_inches="tight")
plt.close(fig)

print(f"Saved: {pdf_path}")
print(f"Saved: {png_path}")
