#!/usr/bin/env python3
"""Generate combined I-AUC/P-AP comparison table for MVTec-AD."""

import csv
import re

# ============================================================
# 1. Data from temp.md — competitor results
# ============================================================

CLASSES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper"
]

CLASS_ABBR = {
    "bottle": "Btl", "cable": "Cbl", "capsule": "Cap",
    "carpet": "Crp", "grid": "Grd", "hazelnut": "Hzl",
    "leather": "Lth", "metal_nut": "MtN", "pill": "Pil",
    "screw": "Scr", "tile": "Tle", "toothbrush": "TBr",
    "transistor": "Trn", "wood": "Wd", "zipper": "Zip"
}

# Image AUC data (from temp.md)
img_auc = {
    "Joint\_PC": {
        "bottle": 1.000, "cable": 0.977, "capsule": 0.927, "carpet": 1.000,
        "grid": 0.983, "hazelnut": 0.994, "leather": 1.000, "metal_nut": 1.000,
        "pill": 0.948, "screw": 0.920, "tile": 1.000, "toothbrush": 0.969,
        "transistor": 0.958, "wood": 0.997, "zipper": 0.998, "avg": 0.978, "fm": None
    },
    "Joint\_PC(R)": {
        "bottle": 0.998, "cable": 0.936, "capsule": 0.868, "carpet": 1.000,
        "grid": 0.984, "hazelnut": 0.979, "leather": 1.000, "metal_nut": 0.993,
        "pill": 0.921, "screw": 0.653, "tile": 0.998, "toothbrush": 0.975,
        "transistor": 0.832, "wood": 0.995, "zipper": 0.972, "avg": 0.940, "fm": None
    },
    "Joint\_CADIC": {
        "bottle": 1.000, "cable": 0.986, "capsule": 0.921, "carpet": 1.000,
        "grid": 0.987, "hazelnut": 0.990, "leather": 1.000, "metal_nut": 1.000,
        "pill": 0.945, "screw": 0.899, "tile": 1.000, "toothbrush": 0.972,
        "transistor": 0.975, "wood": 0.994, "zipper": 0.996, "avg": 0.978, "fm": None
    },
    "FT\_PatchCore": {
        "bottle": 0.163, "cable": 0.518, "capsule": 0.350, "carpet": 0.968,
        "grid": 0.700, "hazelnut": 0.839, "leather": 0.625, "metal_nut": 0.259,
        "pill": 0.459, "screw": 0.484, "tile": 0.776, "toothbrush": 0.586,
        "transistor": 0.341, "wood": 0.970, "zipper": 0.991, "avg": 0.602, "fm": 0.383
    },
    "FT\_CFA": {
        "bottle": 0.309, "cable": 0.489, "capsule": 0.275, "carpet": 0.834,
        "grid": 0.571, "hazelnut": 0.903, "leather": 0.935, "metal_nut": 0.464,
        "pill": 0.528, "screw": 0.528, "tile": 0.763, "toothbrush": 0.519,
        "transistor": 0.320, "wood": 0.923, "zipper": 0.984, "avg": 0.623, "fm": 0.361
    },
    "FT\_SimpleNet": {
        "bottle": 0.938, "cable": 0.560, "capsule": 0.519, "carpet": 0.736,
        "grid": 0.592, "hazelnut": 0.859, "leather": 0.749, "metal_nut": 0.710,
        "pill": 0.701, "screw": 0.599, "tile": 0.654, "toothbrush": 0.422,
        "transistor": 0.669, "wood": 0.908, "zipper": 0.996, "avg": 0.708, "fm": 0.211
    },
    "FT\_RD4AD": {
        "bottle": 0.401, "cable": 0.538, "capsule": 0.475, "carpet": 0.583,
        "grid": 0.558, "hazelnut": 0.909, "leather": 0.596, "metal_nut": 0.623,
        "pill": 0.479, "screw": 0.596, "tile": 0.715, "toothbrush": 0.397,
        "transistor": 0.385, "wood": 0.700, "zipper": 0.987, "avg": 0.596, "fm": 0.393
    },
    "CFRDC": {
        "bottle": 0.996, "cable": 0.900, "capsule": 0.785, "carpet": 0.997,
        "grid": 0.980, "hazelnut": 0.994, "leather": 1.000, "metal_nut": 0.995,
        "pill": 0.933, "screw": 0.711, "tile": 0.991, "toothbrush": 0.933,
        "transistor": 0.997, "wood": 0.982, "zipper": 0.984, "avg": 0.945, "fm": None
    },
    "IUF": {
        "bottle": 0.909, "cable": 0.541, "capsule": 0.520, "carpet": 0.996,
        "grid": 0.695, "hazelnut": 0.875, "leather": 0.997, "metal_nut": 0.643,
        "pill": 0.547, "screw": 0.646, "tile": 0.940, "toothbrush": 0.711,
        "transistor": 0.660, "wood": 0.953, "zipper": 0.795, "avg": 0.762, "fm": 0.067
    },
    "ReplayCAD": {
        "bottle": 0.990, "cable": 0.957, "capsule": 0.747, "carpet": 0.980,
        "grid": 0.927, "hazelnut": 0.985, "leather": 0.974, "metal_nut": 0.995,
        "pill": 0.944, "screw": 0.795, "tile": 0.999, "toothbrush": 0.981,
        "transistor": 0.957, "wood": 0.984, "zipper": 0.997, "avg": 0.948, "fm": 0.045
    },
    "DNE": {
        "bottle": 0.990, "cable": 0.619, "capsule": 0.609, "carpet": 0.984,
        "grid": 0.998, "hazelnut": 0.924, "leather": 1.000, "metal_nut": 0.989,
        "pill": 0.671, "screw": 0.588, "tile": 0.980, "toothbrush": 0.933,
        "transistor": 0.877, "wood": 0.930, "zipper": 0.958, "avg": 0.870, "fm": 0.116
    },
    "UCAD": {
        "bottle": 1.000, "cable": 0.751, "capsule": 0.866, "carpet": 0.965,
        "grid": 0.944, "hazelnut": 0.994, "leather": 1.000, "metal_nut": 0.988,
        "pill": 0.894, "screw": 0.739, "tile": 0.998, "toothbrush": 1.000,
        "transistor": 0.874, "wood": 0.995, "zipper": 0.938, "avg": 0.930, "fm": 0.010
    },
    "DFM": {
        "bottle": 0.997, "cable": 0.948, "capsule": 0.996, "carpet": 0.999,
        "grid": 0.990, "hazelnut": 0.977, "leather": 1.000, "metal_nut": 1.000,
        "pill": 0.983, "screw": 0.765, "tile": 0.982, "toothbrush": 0.997,
        "transistor": 0.932, "wood": 0.986, "zipper": 0.987, "avg": 0.969, "fm": 0.015
    },
    "CADIC": {
        "bottle": 1.000, "cable": 0.982, "capsule": 0.877, "carpet": 0.996,
        "grid": 0.983, "hazelnut": 0.994, "leather": 1.000, "metal_nut": 1.000,
        "pill": 0.942, "screw": 0.906, "tile": 0.995, "toothbrush": 0.954,
        "transistor": 0.968, "wood": 0.994, "zipper": 0.990, "avg": 0.972, "fm": 0.011
    },
}

# Pixel AP data (from temp.md)
pix_ap = {
    "Joint\_PC": {
        "bottle": 0.820, "cable": 0.514, "capsule": 0.525, "carpet": 0.770,
        "grid": 0.300, "hazelnut": 0.728, "leather": 0.224, "metal_nut": 0.892,
        "pill": 0.811, "screw": 0.336, "tile": 0.620, "toothbrush": 0.527,
        "transistor": 0.637, "wood": 0.683, "zipper": 0.531, "avg": 0.594, "fm": None
    },
    "Joint\_PC(R)": {
        "bottle": 0.826, "cable": 0.505, "capsule": 0.510, "carpet": 0.766,
        "grid": 0.293, "hazelnut": 0.712, "leather": 0.230, "metal_nut": 0.862,
        "pill": 0.785, "screw": 0.157, "tile": 0.664, "toothbrush": 0.561,
        "transistor": 0.515, "wood": 0.641, "zipper": 0.565, "avg": 0.573, "fm": None
    },
    "Joint\_CADIC": {
        "bottle": 0.815, "cable": 0.510, "capsule": 0.519, "carpet": 0.754,
        "grid": 0.292, "hazelnut": 0.744, "leather": 0.210, "metal_nut": 0.886,
        "pill": 0.815, "screw": 0.307, "tile": 0.630, "toothbrush": 0.530,
        "transistor": 0.650, "wood": 0.675, "zipper": 0.528, "avg": 0.591, "fm": None
    },
    "FT\_PatchCore": {
        "bottle": 0.048, "cable": 0.029, "capsule": 0.035, "carpet": 0.552,
        "grid": 0.003, "hazelnut": 0.338, "leather": 0.279, "metal_nut": 0.248,
        "pill": 0.051, "screw": 0.008, "tile": 0.249, "toothbrush": 0.034,
        "transistor": 0.079, "wood": 0.304, "zipper": 0.595, "avg": 0.190, "fm": 0.371
    },
    "FT\_CFA": {
        "bottle": 0.068, "cable": 0.056, "capsule": 0.050, "carpet": 0.271,
        "grid": 0.004, "hazelnut": 0.341, "leather": 0.393, "metal_nut": 0.255,
        "pill": 0.080, "screw": 0.015, "tile": 0.155, "toothbrush": 0.053,
        "transistor": 0.056, "wood": 0.281, "zipper": 0.573, "avg": 0.177, "fm": 0.083
    },
    "FT\_SimpleNet": {
        "bottle": 0.108, "cable": 0.045, "capsule": 0.029, "carpet": 0.018,
        "grid": 0.004, "hazelnut": 0.029, "leather": 0.006, "metal_nut": 0.227,
        "pill": 0.077, "screw": 0.004, "tile": 0.082, "toothbrush": 0.046,
        "transistor": 0.049, "wood": 0.037, "zipper": 0.139, "avg": 0.060, "fm": 0.069
    },
    "FT\_RD4AD": {
        "bottle": 0.055, "cable": 0.040, "capsule": 0.064, "carpet": 0.212,
        "grid": 0.005, "hazelnut": 0.384, "leather": 0.116, "metal_nut": 0.247,
        "pill": 0.061, "screw": 0.015, "tile": 0.193, "toothbrush": 0.034,
        "transistor": 0.059, "wood": 0.097, "zipper": 0.562, "avg": 0.143, "fm": 0.425
    },
    "CFRDC": {
        "bottle": 0.737, "cable": 0.518, "capsule": 0.425, "carpet": 0.506,
        "grid": 0.243, "hazelnut": 0.556, "leather": 0.372, "metal_nut": 0.666,
        "pill": 0.417, "screw": 0.125, "tile": 0.454, "toothbrush": 0.417,
        "transistor": 0.710, "wood": 0.380, "zipper": 0.390, "avg": 0.461, "fm": None
    },
    "IUF": {
        "bottle": 0.289, "cable": 0.054, "capsule": 0.040, "carpet": 0.440,
        "grid": 0.084, "hazelnut": 0.301, "leather": 0.330, "metal_nut": 0.142,
        "pill": 0.048, "screw": 0.012, "tile": 0.310, "toothbrush": 0.049,
        "transistor": 0.065, "wood": 0.326, "zipper": 0.080, "avg": 0.171, "fm": 0.059
    },
    "ReplayCAD": {
        "bottle": 0.710, "cable": 0.369, "capsule": 0.337, "carpet": 0.652,
        "grid": 0.338, "hazelnut": 0.635, "leather": 0.587, "metal_nut": 0.656,
        "pill": 0.698, "screw": 0.329, "tile": 0.531, "toothbrush": 0.576,
        "transistor": 0.605, "wood": 0.500, "zipper": 0.539, "avg": 0.537, "fm": 0.055
    },
    # DNE has no Pixel AP data
    "DNE": None,
    "UCAD": {
        "bottle": 0.752, "cable": 0.290, "capsule": 0.349, "carpet": 0.622,
        "grid": 0.187, "hazelnut": 0.506, "leather": 0.333, "metal_nut": 0.775,
        "pill": 0.634, "screw": 0.214, "tile": 0.549, "toothbrush": 0.298,
        "transistor": 0.398, "wood": 0.535, "zipper": 0.398, "avg": 0.456, "fm": 0.013
    },
    "DFM": {
        "bottle": 0.768, "cable": 0.506, "capsule": 0.241, "carpet": 0.771,
        "grid": 0.228, "hazelnut": 0.479, "leather": 0.432, "metal_nut": 0.690,
        "pill": 0.576, "screw": 0.242, "tile": 0.623, "toothbrush": 0.331,
        "transistor": 0.501, "wood": 0.581, "zipper": 0.511, "avg": 0.511, "fm": 0.013
    },
    "CADIC": {
        "bottle": 0.790, "cable": 0.485, "capsule": 0.506, "carpet": 0.753,
        "grid": 0.276, "hazelnut": 0.749, "leather": 0.191, "metal_nut": 0.880,
        "pill": 0.810, "screw": 0.328, "tile": 0.609, "toothbrush": 0.527,
        "transistor": 0.650, "wood": 0.686, "zipper": 0.517, "avg": 0.584, "fm": 0.015
    },
}

# ============================================================
# 2. Load V48_01 SOTA results for "Ours"
# ============================================================

ours_ia = {}
ours_pa = {}

with open("/Volume/DeCoFlow/logs/V48_01_H04_highres_clean/final_results.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        cls = row["Class Name"]
        ia = float(row["Image AUC"].rstrip("*"))
        pa = float(row["Pixel AP"].rstrip("*"))
        if cls == "Overall":
            ours_ia["avg"] = ia
            ours_pa["avg"] = pa
        else:
            ours_ia[cls] = ia
            ours_pa[cls] = pa

# Add to data dicts
img_auc["\\textbf{Ours}"] = {**ours_ia, "fm": 0.0}
pix_ap["\\textbf{Ours}"] = {**ours_pa, "fm": 0.0}

# ============================================================
# 3. Method ordering and grouping
# ============================================================

METHOD_ORDER = [
    # Joint upper bounds
    "Joint\_PC", "Joint\_PC(R)", "Joint\_CADIC",
    # FT baselines
    "FT\_PatchCore", "FT\_CFA", "FT\_SimpleNet", "FT\_RD4AD",
    # CL methods
    "CFRDC", "IUF", "ReplayCAD", "DNE", "UCAD", "DFM", "CADIC",
    # Ours
    "\\textbf{Ours}",
]

# Divider positions (insert \midrule after these indices)
DIVIDERS_AFTER = [2, 6, 13]  # after Joint_CADIC, FT_RD4AD, CADIC

# ============================================================
# 4. Find best CL method values per column for bolding
# ============================================================

CL_METHODS = ["CFRDC", "IUF", "ReplayCAD", "DNE", "UCAD", "DFM", "CADIC", "\\textbf{Ours}"]

def find_best_per_col(data_dict, keys, methods):
    """Find the best (max) value per column among given methods."""
    best = {}
    for k in keys:
        vals = []
        for m in methods:
            if data_dict.get(m) is not None and k in data_dict[m] and data_dict[m][k] is not None:
                vals.append((data_dict[m][k], m))
        if vals:
            max_val = max(v[0] for v in vals)
            best[k] = max_val
    return best

col_keys = CLASSES + ["avg"]
best_ia = find_best_per_col(img_auc, col_keys, CL_METHODS)
best_pa = find_best_per_col(pix_ap, col_keys, CL_METHODS)

# For FM, best is lowest (excluding None)
def find_best_fm(data_dict, methods):
    vals = []
    for m in methods:
        if data_dict.get(m) is not None and data_dict[m].get("fm") is not None:
            vals.append(data_dict[m]["fm"])
    return min(vals) if vals else None

best_fm_ia = find_best_fm(img_auc, CL_METHODS)
best_fm_pa = find_best_fm(pix_ap, CL_METHODS)

# ============================================================
# 5. Format helpers
# ============================================================

def fmt_pct(val):
    """Format a [0,1] value as percentage with 1 decimal."""
    if val is None:
        return "--"
    return f"{val*100:.1f}"

def fmt_cell(ia_val, pa_val, is_best_ia=False, is_best_pa=False):
    """Format a combined I-AUC/P-AP cell."""
    ia_str = fmt_pct(ia_val)
    pa_str = fmt_pct(pa_val)

    if is_best_ia and ia_str != "--":
        ia_str = f"\\textbf{{{ia_str}}}"
    if is_best_pa and pa_str != "--":
        pa_str = f"\\textbf{{{pa_str}}}"

    return f"{ia_str}\\,/\\,{pa_str}"

def fmt_fm(ia_fm, pa_fm, is_best_ia=False, is_best_pa=False):
    """Format FM cell."""
    if ia_fm is None and pa_fm is None:
        return "--"

    ia_str = fmt_pct(ia_fm) if ia_fm is not None else "--"
    pa_str = fmt_pct(pa_fm) if pa_fm is not None else "--"

    if is_best_ia and ia_str != "--":
        ia_str = f"\\textbf{{{ia_str}}}"
    if is_best_pa and pa_str != "--":
        pa_str = f"\\textbf{{{pa_str}}}"

    return f"{ia_str}\\,/\\,{pa_str}"

# ============================================================
# 6. Generate LaTeX
# ============================================================

ncols = 15 + 2 + 1  # 15 classes + avg + fm + method
col_spec = "l" + " c" * 17

lines = []
lines.append(r"\begin{table}[t]")
lines.append(r"\centering")
lines.append(r"\caption{Comparison with state-of-the-art continual learning methods on MVTec-AD (15-class, 1$\times$1 CL scenario). Each cell: I-AUC\,(\%)\,/\,P-AP\,(\%). Best continual learning results per column in \textbf{bold}. Joint methods serve as upper bounds. FM = Forgetting Measure ($\downarrow$).}")
lines.append(r"\label{tab:main_mvtec}")
lines.append(r"\resizebox{\textwidth}{!}{%")
lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
lines.append(r"\toprule")

# Header
header_parts = ["Method"]
for c in CLASSES:
    header_parts.append(CLASS_ABBR[c])
header_parts.append("Avg.")
header_parts.append("FM")
lines.append(" & ".join(header_parts) + r" \\")
lines.append(r"\midrule")

for idx, method in enumerate(METHOD_ORDER):
    # Get data
    ia_data = img_auc.get(method, {})
    pa_data = pix_ap.get(method)

    is_cl = method in CL_METHODS

    cells = [method]

    for c in CLASSES:
        ia_val = ia_data.get(c) if ia_data else None
        pa_val = pa_data.get(c) if pa_data is not None else None

        is_best_ia = is_cl and ia_val is not None and abs(ia_val - best_ia.get(c, -1)) < 1e-6
        is_best_pa = is_cl and pa_val is not None and abs(pa_val - best_pa.get(c, -1)) < 1e-6

        cells.append(fmt_cell(ia_val, pa_val, is_best_ia, is_best_pa))

    # Average
    ia_avg = ia_data.get("avg") if ia_data else None
    pa_avg = pa_data.get("avg") if pa_data is not None else None
    is_best_ia_avg = is_cl and ia_avg is not None and abs(ia_avg - best_ia.get("avg", -1)) < 1e-6
    is_best_pa_avg = is_cl and pa_avg is not None and abs(pa_avg - best_pa.get("avg", -1)) < 1e-6
    cells.append(fmt_cell(ia_avg, pa_avg, is_best_ia_avg, is_best_pa_avg))

    # FM
    ia_fm = ia_data.get("fm") if ia_data else None
    pa_fm = pa_data.get("fm") if pa_data is not None else None
    is_best_fm_i = is_cl and ia_fm is not None and best_fm_ia is not None and abs(ia_fm - best_fm_ia) < 1e-6
    is_best_fm_p = is_cl and pa_fm is not None and best_fm_pa is not None and abs(pa_fm - best_fm_pa) < 1e-6
    cells.append(fmt_fm(ia_fm, pa_fm, is_best_fm_i, is_best_fm_p))

    lines.append(" & ".join(cells) + r" \\")

    if idx in DIVIDERS_AFTER:
        lines.append(r"\midrule")

lines.append(r"\bottomrule")
lines.append(r"\end{tabular}%")
lines.append(r"}")
lines.append(r"\end{table}")

result = "\n".join(lines)
print(result)

# Also save to file
with open("/Volume/DeCoFlow/Paper_works/latex/table_mvtec_combined.tex", "w") as f:
    f.write(result)
    f.write("\n")

print("\n\n=== Saved to Paper_works/latex/table_mvtec_combined.tex ===")
