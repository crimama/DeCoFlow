#!/usr/bin/env python
"""
DCL-ACL Functional Division of Labor — DCL-favorable visualization.

3-panel figure:
  (a) Cumulative |log|det J|| — shows DCL does 68% of total transformation
  (b) Per-block transformation intensity — DCL blocks are individually stronger
  (c) Before/After summary — DCL prepares, ACL finalizes

Usage:
  python scripts/plot_dcl_acl_division.py
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

block_labels = data["block_labels"]  # Input, DCL1..6, ACL1, ACL2
classes = data["classes"]
qq_mean = np.array(data["metrics"]["qq_correlation"]["mean"])
qq_std = np.array(data["metrics"]["qq_correlation"]["std"])
offdiag_mean = np.array(data["metrics"]["offdiag_cov_norm"]["mean"])
offdiag_std = np.array(data["metrics"]["offdiag_cov_norm"]["std"])
logdet_mean = np.array(data["metrics"]["abs_logdet"]["mean"])
logdet_std = np.array(data["metrics"]["abs_logdet"]["std"])

# Per-class data for thin lines
qq_per_class = data["metrics"]["qq_correlation"]["per_class"]
offdiag_per_class = data["metrics"]["offdiag_cov_norm"]["per_class"]
logdet_per_class = data["metrics"]["abs_logdet"]["per_class"]

# Colors
DCL_COLOR = "#2563EB"    # Blue
ACL_COLOR = "#DC2626"    # Red
GRAY = "#9CA3AF"
BG_DCL = "#DBEAFE"       # Light blue
BG_ACL = "#FEE2E2"       # Light red

# =====================================================================
# Figure: 3 panels
# =====================================================================
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), dpi=300)

# Block indices: 0=Input, 1-6=DCL1-6, 7=ACL1, 8=ACL2
dcl_indices = list(range(1, 7))  # DCL1-DCL6
acl_indices = [7, 8]              # ACL1, ACL2

# ---- Panel (a): Cumulative log-det ----
ax = axes[0]

# Cumulative per class (thin gray)
for cls in classes:
    cls_logdet = np.array(logdet_per_class[cls])
    cum = np.cumsum(cls_logdet)
    ax.plot(range(len(cum)), cum, color=GRAY, alpha=0.2, linewidth=0.6)

# Cumulative mean
cum_mean = np.cumsum(logdet_mean)
total = cum_mean[-1]

# Fill DCL region and ACL region
ax.fill_between(range(0, 8), 0, cum_mean[:8], alpha=0.15, color=DCL_COLOR, label=None)
ax.fill_between(range(7, 9), cum_mean[6:8], cum_mean[7:9], alpha=0.15, color=ACL_COLOR, label=None)

ax.plot(range(len(cum_mean)), cum_mean, color="black", linewidth=2.0, marker="o",
        markersize=5, markeredgecolor="white", markeredgewidth=0.8, zorder=5)

# Mark DCL total and ACL total
dcl_total = cum_mean[6]
acl_total = total - dcl_total
dcl_pct = dcl_total / total * 100
acl_pct = acl_total / total * 100

# Annotate percentages
ax.annotate(f"DCL: {dcl_pct:.0f}%", xy=(3, cum_mean[3]), xytext=(1.5, cum_mean[6] * 0.55),
            fontsize=11, fontweight="bold", color=DCL_COLOR,
            arrowprops=dict(arrowstyle='->', color=DCL_COLOR, lw=1.2))
ax.annotate(f"ACL: {acl_pct:.0f}%", xy=(7.5, (cum_mean[7] + total) / 2),
            xytext=(6.0, total * 0.92),
            fontsize=11, fontweight="bold", color=ACL_COLOR,
            arrowprops=dict(arrowstyle='->', color=ACL_COLOR, lw=1.2))

# Dashed line at DCL/ACL boundary
ax.axvline(x=6.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

ax.set_xticks(range(len(block_labels)))
ax.set_xticklabels(block_labels, fontsize=7.5, rotation=30, ha="right")
ax.set_ylabel("Cumulative |log|det J||", fontsize=10)
ax.set_title("(a) Cumulative Transformation", fontsize=11, fontweight="bold")
ax.set_xlim(-0.3, 8.3)
ax.set_ylim(bottom=0)

# ---- Panel (b): Per-block intensity bar chart ----
ax = axes[1]

# Per-block mean log-det (skip Input which is 0)
block_names = block_labels[1:]  # DCL1..DCL6, ACL1, ACL2
block_logdet = logdet_mean[1:]
block_logdet_std = logdet_std[1:]

colors = [DCL_COLOR] * 6 + [ACL_COLOR] * 2
bars = ax.bar(range(len(block_names)), block_logdet, color=colors, alpha=0.8,
              edgecolor="white", linewidth=0.5)
ax.errorbar(range(len(block_names)), block_logdet, yerr=block_logdet_std,
            fmt="none", ecolor="gray", elinewidth=0.8, capsize=2)

# Add value labels on top
for i, (v, s) in enumerate(zip(block_logdet, block_logdet_std)):
    ax.text(i, v + s + 5, f"{v:.0f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

# DCL per-block average vs ACL per-block average
dcl_avg = np.mean(block_logdet[:6])
acl_avg = np.mean(block_logdet[6:])
ax.axhline(y=dcl_avg, color=DCL_COLOR, linestyle=":", linewidth=1.0, alpha=0.7)
ax.axhline(y=acl_avg, color=ACL_COLOR, linestyle=":", linewidth=1.0, alpha=0.7)

ax.text(5.8, dcl_avg + 5, f"DCL avg: {dcl_avg:.0f}", fontsize=8, color=DCL_COLOR, ha="right")
ax.text(7.2, acl_avg + 5, f"ACL avg: {acl_avg:.0f}", fontsize=8, color=ACL_COLOR, ha="center")

ax.set_xticks(range(len(block_names)))
ax.set_xticklabels(block_names, fontsize=7.5, rotation=30, ha="right")
ax.set_ylabel("|log|det J|| per block", fontsize=10)
ax.set_title("(b) Per-Block Transformation", fontsize=11, fontweight="bold")

# Legend
dcl_patch = mpatches.Patch(color=DCL_COLOR, alpha=0.8, label="DCL")
acl_patch = mpatches.Patch(color=ACL_COLOR, alpha=0.8, label="ACL")
ax.legend(handles=[dcl_patch, acl_patch], loc="upper right", fontsize=9)

# ---- Panel (c): Before/After summary — multi-metric ----
ax = axes[2]

# 3 stages: Input, After DCL, After ACL
stages = ["Input", "After\nDCL", "After\nACL"]
x_pos = np.array([0, 1, 2])

# Q-Q correlation
qq_vals = [qq_mean[0], qq_mean[6], qq_mean[8]]  # Input, DCL6, ACL2
qq_stds = [qq_std[0], qq_std[6], qq_std[8]]

# Off-diagonal (normalize to [0,1] range for visual, invert so lower = better)
# Use raw values but on secondary axis
offdiag_vals = [offdiag_mean[0], offdiag_mean[6], offdiag_mean[8]]
offdiag_stds = [offdiag_std[0], offdiag_std[6], offdiag_std[8]]

# Plot Q-Q as bars
width = 0.35
bars_qq = ax.bar(x_pos - width/2, qq_vals, width, color=[GRAY, DCL_COLOR, ACL_COLOR],
                 alpha=0.8, edgecolor="white", linewidth=0.5, label="Q-Q Corr. (↑)")
ax.errorbar(x_pos - width/2, qq_vals, yerr=qq_stds, fmt="none", ecolor="gray",
            elinewidth=0.8, capsize=3)

# Add value labels
for i, (v, s) in enumerate(zip(qq_vals, qq_stds)):
    ax.text(x_pos[i] - width/2, v + s + 0.003, f"{v:.3f}", ha="center", va="bottom",
            fontsize=8, fontweight="bold")

# Plot off-diagonal on secondary y-axis
ax2 = ax.twinx()
bars_od = ax2.bar(x_pos + width/2, offdiag_vals, width,
                  color=[GRAY, DCL_COLOR, ACL_COLOR],
                  alpha=0.35, edgecolor="gray", linewidth=0.5,
                  hatch="///", label="Off-diag (↓)")
ax2.errorbar(x_pos + width/2, offdiag_vals, yerr=offdiag_stds, fmt="none",
             ecolor="gray", elinewidth=0.8, capsize=3)

for i, (v, s) in enumerate(zip(offdiag_vals, offdiag_stds)):
    ax2.text(x_pos[i] + width/2, v + s + 0.01, f"{v:.3f}", ha="center", va="bottom",
             fontsize=8, style="italic")

ax.set_xticks(x_pos)
ax.set_xticklabels(stages, fontsize=10)
ax.set_ylabel("Q-Q Correlation (↑)", fontsize=10, color="black")
ax.set_ylim(0.85, 1.02)
ax2.set_ylabel("Off-diagonal Norm (↓)", fontsize=10, color="gray")
ax2.set_ylim(0, 0.5)

ax.set_title("(c) Statistical Quality", fontsize=11, fontweight="bold")

# Combined legend
from matplotlib.lines import Line2D
legend_elements = [
    mpatches.Patch(facecolor=DCL_COLOR, alpha=0.8, label="Q-Q Corr. (↑)"),
    mpatches.Patch(facecolor=DCL_COLOR, alpha=0.35, hatch="///", edgecolor="gray", label="Off-diag Norm (↓)"),
]
ax.legend(handles=legend_elements, loc="lower left", fontsize=8)

plt.tight_layout()

# Save
import os
pdf_path = os.path.join(OUT_DIR, "dcl_acl_division.pdf")
png_path = os.path.join(OUT_DIR, "dcl_acl_division.png")
fig.savefig(pdf_path, bbox_inches="tight")
fig.savefig(png_path, bbox_inches="tight")
plt.close(fig)

print(f"Saved: {pdf_path}")
print(f"Saved: {png_path}")

# Summary statistics
print("\n=== Summary ===")
print(f"DCL cumulative log-det: {dcl_total:.1f} ({dcl_pct:.1f}%)")
print(f"ACL cumulative log-det: {acl_total:.1f} ({acl_pct:.1f}%)")
print(f"DCL per-block avg: {dcl_avg:.1f}")
print(f"ACL per-block avg: {acl_avg:.1f}")
print(f"Q-Q: Input={qq_vals[0]:.4f} → After DCL={qq_vals[1]:.4f} → After ACL={qq_vals[2]:.4f}")
print(f"Off-diag: Input={offdiag_vals[0]:.4f} → After DCL={offdiag_vals[1]:.4f} → After ACL={offdiag_vals[2]:.4f}")
