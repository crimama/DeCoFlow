"""
Generate scatter_efficiency.pdf — Total System Size vs Performance scatter plot.
X-axis: Total system parameters after T tasks (M, log scale)
        Includes ALL models required (backbone, AD model, external generative models)
Y-axis: I-AUC (%)
Star marker for FM=0.0% methods, circle for FM>0%

Baseline parameter analysis (computed from code in Baseline/):
  - DNE:       ViT-B/16 (86.6M) + 15×Gaussian(591K) = 95.5M
  - UCAD:      2×ViT-B/16(173M) + 15×(prompt+memory)(787K) = 185M
  - IUF:       EfficientNet-B4(19M) + UniAD(9.1M) = 28.1M
  - ReplayCAD: EfficientNet-B4(19M) + UniAD(9.1M) + LDM(~1B) = 1,028M
  - CDAD:      SD v1.5(1067M) + AMN(11M) + RN50(26M) + 15×GPM(4M) = 1,164M
  - DeCoFlow:  WRN50-2+NF (104.5M) + 15×LoRA(1.93M) = 133.4M (from checkpoint)
"""
import sys
sys.path.insert(0, '/Volume/DeCoFlow')

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from collections import defaultdict

# ============================================================
# 1. DeCoFlow — exact from checkpoint
# ============================================================
ckpt_dir = '/Volume/DeCoFlow/logs/V48_01_H04_highres_clean/checkpoints/task_14'
state_dict = torch.load(f'{ckpt_dir}/nf_model.pth', map_location='cpu', weights_only=False)

nf_base_params = 0
per_task_params = defaultdict(lambda: {'lora': 0, 'acb': 0, 'bias': 0, 'tsa': 0, 'other': 0})

for key, val in state_dict.items():
    n = val.numel()
    if '.lora_A.' in key or '.lora_B.' in key:
        parts = key.split('.')
        for i, p in enumerate(parts):
            if p in ('lora_A', 'lora_B') and i + 1 < len(parts):
                tid = parts[i + 1]
                per_task_params[tid]['lora'] += n
                break
    elif 'acb_blocks' in key or 'dia_' in key:
        parts = key.split('.')
        tid = None
        for i, p in enumerate(parts):
            if p == 'task_params' and i + 1 < len(parts):
                tid = parts[i + 1]
                break
        if tid is not None and tid.isdigit():
            per_task_params[tid]['acb'] += n
        else:
            nf_base_params += n
    elif 'task_bias' in key:
        parts = key.split('.')
        tid = None
        for p in parts:
            if p.isdigit():
                tid = p
                break
        if tid is not None:
            per_task_params[tid]['bias'] += n
        else:
            nf_base_params += n
    elif 'tsa' in key or 'task_input_adapter' in key:
        parts = key.split('.')
        tid = None
        for p in parts:
            if p.isdigit():
                tid = p
                break
        if tid is not None:
            per_task_params[tid]['tsa'] += n
        else:
            nf_base_params += n
    else:
        nf_base_params += n

backbone_params = 68_883_776  # WRN50_2
total_frozen = backbone_params + nf_base_params

task_totals = []
for tid in sorted(per_task_params.keys(), key=lambda x: int(x)):
    d = per_task_params[tid]
    total = sum(d.values())
    task_totals.append(total)
avg_per_task = np.mean(task_totals)

T = 15  # number of tasks
decoflow_total = total_frozen + T * avg_per_task  # frozen + 15 × per-task

print(f"=== DeCoFlow (from checkpoint) ===")
print(f"  Frozen: {total_frozen:,} ({total_frozen/1e6:.1f}M)")
print(f"  Per-task: {avg_per_task:,.0f} ({avg_per_task/1e6:.3f}M)")
print(f"  Total after {T} tasks: {decoflow_total:,.0f} ({decoflow_total/1e6:.1f}M)")

# ============================================================
# 2. Baseline parameter analysis (computed from code in Baseline/)
# ============================================================
print(f"\n=== Baseline Parameter Analysis ===")

# --- DNE (Baseline/Continual_Anomaly_Detection) ---
# Architecture: ViT-B/16 backbone (fine-tuned) + DNE Gaussian density estimation
# Source: models/vit.py (ViT-B/16), methods/dne.py (GaussianDensityTorch with Ledoit-Wolf)
# Per-task: stores mean(768) + covariance(768×768) for Mahalanobis scoring
vit_b16_params = 86_567_680  # ViT-B/16 standard param count
dne_frozen = vit_b16_params
dne_mean = 768
dne_cov = 768 * 768  # full covariance matrix (Ledoit-Wolf shrinkage)
dne_per_task = dne_mean + dne_cov  # = 590,592
dne_total = dne_frozen + T * dne_per_task
print(f"\nDNE:")
print(f"  Backbone: ViT-B/16 = {dne_frozen:,} ({dne_frozen/1e6:.1f}M)")
print(f"  Per-task: mean({dne_mean}) + cov({dne_cov:,}) = {dne_per_task:,}")
print(f"  Total after {T} tasks: {dne_total:,.0f} ({dne_total/1e6:.1f}M)")

# --- UCAD (Baseline/UCAD) ---
# Architecture: 2×ViT-B/16 (model + prompt_model, per paper configuration)
#   + E-Prompt (prefix-tuning prompts per task) + PatchCore memory bank
# Source: patchcore/patchcore.py, patchcore/prompt.py
# Paper uses ViT-B/16 as feature backbone (not WRN50-2 from code default)
# Prompt model: ViT-B/16 with E-Prompt (pool_size=1, length=1, 12 layers, prefix-tune K+V)
# Per-task E-Prompt: 12 layers × 2(KV) × 768 = 18,432
# Per-task prompt key: 1 × 768 = 768
# Per-task PatchCore coreset: ~1000 features × 768-d = ~768K (ViT-B/16 features)
ucad_backbones = 2 * vit_b16_params  # 2×ViT-B/16 (model + prompt_model)
ucad_prompt = 12 * 2 * 768  # E-Prompt prefix-tune: 18,432
ucad_key = 768  # prompt key
ucad_memory = 1000 * 768  # PatchCore coreset with ViT-B/16 768-d features: ~768,000
ucad_per_task = ucad_prompt + ucad_key + ucad_memory  # ~787,200
ucad_total = ucad_backbones + T * ucad_per_task
print(f"\nUCAD:")
print(f"  Backbones: 2×ViT-B/16 = {ucad_backbones/1e6:.1f}M")
print(f"  Per-task: prompt({ucad_prompt:,}) + key({ucad_key}) + memory({ucad_memory:,}) = {ucad_per_task:,}")
print(f"  Total after {T} tasks: {ucad_total:,.0f} ({ucad_total/1e6:.1f}M)")

# --- IUF (Baseline/IUF) --- ECCV 2024
# Architecture: EfficientNet-B4 (frozen backbone) + MFCN neck (no params)
#   + UniAD reconstruction (transformer encoder-decoder)
# Source: config_c1.yaml, models/backbones/efficientnet_b4, models/reconstructions/UniAD
# Config: hidden_dim=256, 4 encoder + 4 decoder layers, nhead=8, dim_feedforward=1024
# CL strategy: ConstrainedSGD (gradient projection, stored projections for training only)
# At inference: only the shared model is needed
efficientnet_b4_params = 19_000_000  # EfficientNet-B4 (frozen)
# UniAD: input/output proj + learned pos embed + 4 enc layers + 4 dec layers
# Encoder layer: self-attn(4×256²) + FFN(256×1024 + 1024×256) + 2×LN(256)
# Decoder layer: self-attn + cross-attn + FFN + 3×LN + 196 learned queries
uniad_params = 9_140_000  # UniAD + optional ViT classifier
iuf_total = efficientnet_b4_params + uniad_params
# ConstrainedSGD projections: training-time only, but stored for CL
# Conservative estimate: ~0.3M per task (reduced-rank SVD projections)
iuf_cl_per_task = 300_000
iuf_total_with_cl = iuf_total + T * iuf_cl_per_task
print(f"\nIUF:")
print(f"  EfficientNet-B4 (frozen): {efficientnet_b4_params/1e6:.1f}M")
print(f"  UniAD reconstruction: {uniad_params/1e6:.1f}M")
print(f"  CL overhead (ConstrainedSGD): ~{iuf_cl_per_task/1e6:.2f}M/task")
print(f"  Total after {T} tasks: {iuf_total_with_cl:,.0f} ({iuf_total_with_cl/1e6:.1f}M)")

# --- ReplayCAD (Baseline/ReplayCAD) --- IJCAI 2025
# Architecture: EfficientNet-B4 (frozen) + UniAD for AD model (same as IUF)
#   + LDM/Stable Diffusion (~1B) for textual inversion replay data generation
# Source: configs/invad/invad_mvtec.py, model/invad.py
# AD model: EfficientNet-B4 + UniAD (same architecture as IUF)
# Replay: SD/LDM generates synthetic images via textual inversion per class
# Per-task: textual inversion embedding (768 params) — negligible
# Note: SD/LDM is REQUIRED for the CL strategy (generating replay data for new tasks)
replaycad_ad_model = efficientnet_b4_params + uniad_params  # same as IUF
sd_ldm_for_replay = 1_000_000_000  # Stable Diffusion / LDM required for replay generation
replaycad_text_embed = 768  # textual inversion embedding per task
replaycad_per_task = replaycad_text_embed
replaycad_total = replaycad_ad_model + sd_ldm_for_replay + T * replaycad_per_task
print(f"\nReplayCAD:")
print(f"  AD model: EfficientNet-B4+UniAD = {replaycad_ad_model/1e6:.1f}M")
print(f"  + SD/LDM for replay: {sd_ldm_for_replay/1e6:.0f}M")
print(f"  Per-task: textual_inversion({replaycad_text_embed})")
print(f"  Total after {T} tasks: {replaycad_total:,.0f} ({replaycad_total/1e6:.0f}M)")

# --- CDAD / One-for-More (Baseline/CDAD) --- CVPR 2025
# Architecture: Stable Diffusion v1.5 (UNet + CLIP + VAE) + AMN ControlNet + ResNet50
# Source: models/cdad_mvtec.yaml, cdm/sd_amn.py, cdm/gpm.py
# CL strategy: GPM (Gradient Projection Memory) via iSVD
# Stores SVD projections of activations per task for gradient orthogonalization
sd_unet = 860_000_000
clip_encoder = 123_000_000
vae_model = 84_000_000
amn_model = 11_000_000  # ControlNet-like: input_hint_block + 8-layer transformer
resnet50_model = 25_600_000
cdad_base_model = sd_unet + clip_encoder + vae_model + amn_model + resnet50_model
cdad_gpm_per_task = 4_000_000  # estimated GPM projection matrices
cdad_total = cdad_base_model + T * cdad_gpm_per_task
print(f"\nCDAD (One-for-More):")
print(f"  Base model: {cdad_base_model/1e6:.0f}M")
print(f"    SD UNet({sd_unet/1e6:.0f}M) + CLIP({clip_encoder/1e6:.0f}M) + VAE({vae_model/1e6:.0f}M)")
print(f"    + AMN({amn_model/1e6:.0f}M) + RN50({resnet50_model/1e6:.1f}M)")
print(f"  Per-task GPM: ~{cdad_gpm_per_task/1e6:.0f}M")
print(f"  Total after {T} tasks: {cdad_total:,.0f} ({cdad_total/1e6:.0f}M)")

# ============================================================
# 3. Compile all data points (from Table 1 of main.tex)
# ============================================================
# Format: (name, total_params_M, i_auc_%, p_ap_%, fm_%p, category)
# Performance values from Paper_works/latex/main.tex Table 1
# DNE has no P-AP reported (--) in the paper
methods = [
    # (name, total_M, I-AUC%, P-AP%, FM, category)
    ('DeCoFlow\n(Ours)',  decoflow_total / 1e6,        98.4, 58.2, 0.0,   'ours'),
    ('DNE',               dne_total / 1e6,             87.0, None, 11.6,  'regularization'),
    ('UCAD',              ucad_total / 1e6,            93.0, 45.6, 1.0,   'prompt'),
    ('IUF',               iuf_total_with_cl / 1e6,    76.2, 17.1, 6.7,   'reconstruction'),
    ('ReplayCAD',         replaycad_total / 1e6,       94.8, 53.7, 4.5,   'replay'),
    ('CDAD',              cdad_total / 1e6,            74.3, 28.0, 20.1,  'reconstruction'),
]

print(f"\n=== Summary for scatter plot ===")
print(f"{'Method':<15} {'Total(M)':<12} {'I-AUC(%)':<10} {'P-AP(%)':<10} {'FM(%p)':<8}")
print("-" * 65)
for name, total_m, iauc, pap, fm, cat in methods:
    pap_str = f"{pap:.1f}" if pap is not None else "--"
    print(f"{name.replace(chr(10),' '):<15} {total_m:<12.1f} {iauc:<10.1f} {pap_str:<10} {fm:<8.1f}")

# ============================================================
# 4. Plot — 1×2 layout: (a) Image AUC, (b) Pixel AP
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

# Color palette
colors = {
    'ours':           '#C0392B',  # dark red
    'regularization': '#1A5276',  # dark blue
    'replay':         '#1E8449',  # dark green
    'prompt':         '#6C3483',  # dark purple
    'reconstruction': '#D35400',  # dark orange
}

from matplotlib.lines import Line2D

def plot_scatter(ax, methods_data, y_key, ylabel, ylim, yticks, annot_cfg):
    """Plot scatter on a given axes."""
    for name, total_m, iauc, pap, fm, cat in methods_data:
        y_val = iauc if y_key == 'iauc' else pap
        if y_val is None:
            continue  # skip methods without this metric
        color = colors[cat]
        if fm == 0.0:
            ax.plot(total_m, y_val,
                    marker='*', markersize=22, color=color,
                    markeredgecolor='black', markeredgewidth=0.8,
                    zorder=10, linestyle='none')
        else:
            size = 80 + fm * 15
            ax.scatter(total_m, y_val, s=size, c=color,
                       edgecolors='black', linewidths=0.8,
                       zorder=8, alpha=0.9)

    # Annotations
    for name, total_m, iauc, pap, fm, cat in methods_data:
        y_val = iauc if y_key == 'iauc' else pap
        if y_val is None:
            continue
        cfg = annot_cfg.get(name, {})
        if cfg:
            ax.annotate(name, xy=(total_m, y_val),
                        xytext=cfg['offset'], textcoords='offset points',
                        fontsize=14, fontweight=cfg['fontweight'], color=cfg['color'],
                        ha=cfg.get('ha', 'center'), va=cfg.get('va', 'top'))

    # Axis formatting
    ax.set_xscale('log')
    ax.set_xlabel('Total System Parameters (M)', fontsize=10, fontweight='medium')
    ax.set_ylabel(ylabel, fontsize=10, fontweight='medium')
    ax.set_xlim(20, 2000)
    ax.set_ylim(ylim)
    ax.set_xticks([30, 50, 100, 200, 500, 1000])
    ax.get_xaxis().set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:g}'))
    ax.tick_params(axis='both', labelsize=9)
    ax.set_yticks(yticks)
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.6)
    ax.set_axisbelow(True)

# --- (a) Image AUC ---
annot_iauc = {
    'DeCoFlow\n(Ours)': dict(offset=(-30, -16), ha='center', va='top',
                              fontweight='bold', color=colors['ours']),
    'DNE':              dict(offset=(-16, 12), ha='right', va='bottom',
                              fontweight='normal', color=colors['regularization']),
    'UCAD':             dict(offset=(14, -14), ha='left', va='top',
                              fontweight='normal', color=colors['prompt']),
    'IUF':              dict(offset=(12, 2), ha='left', va='center',
                              fontweight='normal', color=colors['reconstruction']),
    'ReplayCAD':        dict(offset=(-14, 10), ha='right', va='bottom',
                              fontweight='normal', color=colors['replay']),
    'CDAD':             dict(offset=(-14, 10), ha='right', va='bottom',
                              fontweight='normal', color=colors['reconstruction']),
}
plot_scatter(ax1, methods, 'iauc', 'Image AUC (%)',
             ylim=(68, 102), yticks=range(70, 101, 5), annot_cfg=annot_iauc)
ax1.set_title('(a) Image-level Detection', fontsize=11, fontweight='medium', pad=8)

# --- (b) Pixel AP ---
# DNE has no P-AP so it won't appear; adjust annotations for remaining 5 methods
annot_pap = {
    'DeCoFlow\n(Ours)': dict(offset=(-30, -16), ha='center', va='top',
                              fontweight='bold', color=colors['ours']),
    'UCAD':             dict(offset=(12, -12), ha='left', va='top',
                              fontweight='normal', color=colors['prompt']),
    'IUF':              dict(offset=(12, 2), ha='left', va='center',
                              fontweight='normal', color=colors['reconstruction']),
    'ReplayCAD':        dict(offset=(-14, 10), ha='right', va='bottom',
                              fontweight='normal', color=colors['replay']),
    'CDAD':             dict(offset=(-14, 10), ha='right', va='bottom',
                              fontweight='normal', color=colors['reconstruction']),
}
plot_scatter(ax2, methods, 'pap', 'Pixel AP (%)',
             ylim=(10, 65), yticks=range(10, 61, 10), annot_cfg=annot_pap)
ax2.set_title('(b) Pixel-level Localization', fontsize=11, fontweight='medium', pad=8)

# Shared legend (placed on right panel)
legend_elements = [
    Line2D([0], [0], marker='*', color='w', markerfacecolor='gray',
           markeredgecolor='black', markersize=15, label='FM = 0.0%'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
           markeredgecolor='black', markersize=9, label='FM > 0%'),
]
ax2.legend(handles=legend_elements, loc='lower right', fontsize=9,
           framealpha=0.9, edgecolor='gray')

plt.tight_layout(pad=0.8)

# Save - High resolution for print
out_path = '/Volume/DeCoFlow/Paper_works/figures/scatter_efficiency.pdf'
fig.savefig(out_path, dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"\nSaved: {out_path}")

out_png = '/Volume/DeCoFlow/Paper_works/figures/scatter_efficiency.png'
fig.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"Saved: {out_png}")
