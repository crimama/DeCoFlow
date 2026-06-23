"""
ViT Block Combination Analysis — 6 classes × 다양한 조합
최적 블록 세팅을 찾기 위한 exhaustive 비교
"""
import sys
sys.path.insert(0, '/Volume/DeCoFlow')

import argparse
import torch
import torch.nn.functional as F
import numpy as np
import timm
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score, average_precision_score
from decoflow.data.visa import VISA as VisADataset


def extract_block_features(backbone, images, block_indices, device):
    backbone.eval()
    images = images.to(device)
    features = {}
    hooks = []
    for idx in block_indices:
        def hook_fn(module, input, output, block_idx=idx):
            features[block_idx] = output
        h = backbone.blocks[idx].register_forward_hook(hook_fn)
        hooks.append(h)
    with torch.no_grad():
        _ = backbone(images)
    for h in hooks:
        h.remove()
    return features


def features_to_spatial(feat):
    feat = feat[:, 1:, :]
    B, N, D = feat.shape
    H = W = int(np.sqrt(N))
    return feat.reshape(B, H, W, D)


def evaluate_combo(backbone, block_ids, cls_name, data_path, device, upsample_to=None):
    """Evaluate a block combination on one class."""
    train_dataset = VisADataset(
        root=data_path, class_name=cls_name, train=True,
        img_size=224, crp_size=224, msk_size=256
    )
    test_dataset = VisADataset(
        root=data_path, class_name=cls_name, train=False,
        img_size=224, crp_size=224, msk_size=256
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # All blocks we need
    all_needed = sorted(set(block_ids))

    # Train: compute mean/std
    train_feats = []
    for images, _, _, _, _ in train_loader:
        feats = extract_block_features(backbone, images, all_needed, device)
        combined = torch.stack([features_to_spatial(feats[b]) for b in block_ids], dim=0).mean(dim=0)
        if upsample_to and upsample_to != 14:
            combined = combined.permute(0, 3, 1, 2)
            combined = F.interpolate(combined, size=(upsample_to, upsample_to), mode='bilinear', align_corners=False)
            combined = combined.permute(0, 2, 3, 1)
        train_feats.append(combined.cpu())

    train_all = torch.cat(train_feats, dim=0)
    t_mean = train_all.mean(dim=0, keepdim=True)
    t_std = train_all.std(dim=0, keepdim=True) + 1e-8

    # Test
    all_scores, all_img_scores, all_labels, all_masks = [], [], [], []
    for images, labels, masks, _, _ in test_loader:
        feats = extract_block_features(backbone, images, all_needed, device)
        combined = torch.stack([features_to_spatial(feats[b]) for b in block_ids], dim=0).mean(dim=0)
        if upsample_to and upsample_to != 14:
            combined = combined.permute(0, 3, 1, 2)
            combined = F.interpolate(combined, size=(upsample_to, upsample_to), mode='bilinear', align_corners=False)
            combined = combined.permute(0, 2, 3, 1)

        z = (combined.cpu() - t_mean) / t_std
        scores = (z ** 2).sum(dim=-1)
        scores_up = F.interpolate(
            scores.unsqueeze(1).float(), size=(256, 256), mode='bicubic', align_corners=False
        ).squeeze(1)
        img_scores = scores.reshape(scores.shape[0], -1).max(dim=1)[0]

        all_scores.append(scores_up.numpy())
        all_img_scores.append(img_scores.numpy())
        all_labels.extend(labels.numpy())
        all_masks.extend(masks.numpy())

    all_scores = np.concatenate(all_scores)
    all_img_scores = np.concatenate(all_img_scores)
    all_labels = np.array(all_labels, dtype=bool)
    all_masks = np.squeeze(np.array(all_masks, dtype=bool), axis=1)

    for i in range(all_scores.shape[0]):
        all_scores[i] = gaussian_filter(all_scores[i], sigma=0.5)

    all_scores = np.nan_to_num(all_scores, nan=0.0, posinf=1e6, neginf=0.0)
    all_img_scores = np.nan_to_num(all_img_scores, nan=0.0, posinf=1e6, neginf=0.0)

    try:
        img_auc = roc_auc_score(all_labels, all_img_scores) * 100
    except:
        img_auc = 50.0
    try:
        pixel_ap = average_precision_score(all_masks.flatten(), all_scores.flatten()) * 100
    except:
        pixel_ap = 0.0

    return img_auc, pixel_ap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--data_path', default='/Volume/VisA')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}')
    backbone = timm.create_model('vit_base_patch16_224.augreg2_in21k_ft_in1k', pretrained=True, img_size=224)
    backbone = backbone.to(device)
    backbone.eval()

    classes = ['cashew', 'pcb1', 'macaroni1', 'capsules', 'chewinggum', 'pipe_fryum']

    # Block combinations to test
    combos = {
        # Current
        'current [3,6,9,11]': [3, 6, 9, 11],
        # Shallow focused
        'shallow3 [0,1,2]': [0, 1, 2],
        'shallow4 [0,1,2,3]': [0, 1, 2, 3],
        'shallow+mid [1,2,3,5]': [1, 2, 3, 5],
        # Shallow + deep mix
        'sh+deep [1,2,9,11]': [1, 2, 9, 11],
        'sh+deep [0,2,6,11]': [0, 2, 6, 11],
        'sh+deep [1,3,6,11]': [1, 3, 6, 11],
        'sh+deep [0,1,2,11]': [0, 1, 2, 11],
        # Shallow-heavy
        'sh-heavy [0,1,2,3,9,11]': [0, 1, 2, 3, 9, 11],
        'sh-heavy [0,1,2,6,9,11]': [0, 1, 2, 6, 9, 11],
        # Deep
        'deep [9,10,11]': [9, 10, 11],
        'deep4 [8,9,10,11]': [8, 9, 10, 11],
        # Wide
        'wide [0,3,6,9]': [0, 3, 6, 9],
        'wide [1,4,7,10]': [1, 4, 7, 10],
        # Single best candidates
        'single [1]': [1],
        'single [2]': [2],
    }

    # ============================================================
    # Part 1: Block combo comparison at 14×14
    # ============================================================
    print("=" * 90)
    print("Part 1: Block Combination Comparison (14×14, 6 classes)")
    print("=" * 90)

    all_results = {}
    for combo_name, block_ids in combos.items():
        cls_results = {}
        for cls_name in classes:
            iauc, pap = evaluate_combo(backbone, block_ids, cls_name, args.data_path, device)
            cls_results[cls_name] = (iauc, pap)

        avg_iauc = np.mean([v[0] for v in cls_results.values()])
        avg_pap = np.mean([v[1] for v in cls_results.values()])
        all_results[combo_name] = (avg_iauc, avg_pap, cls_results)

    # Sort by avg P-AP
    sorted_results = sorted(all_results.items(), key=lambda x: x[1][1], reverse=True)

    print(f"\n{'Combo':<30} {'Avg I-AUC':>10} {'Avg P-AP':>10}", end="")
    for c in classes:
        print(f" {c[:6]:>8}", end="")
    print()
    print("-" * (52 + 8*len(classes)))

    for combo_name, (avg_iauc, avg_pap, cls_results) in sorted_results:
        print(f"{combo_name:<30} {avg_iauc:>10.2f} {avg_pap:>10.2f}", end="")
        for c in classes:
            print(f" {cls_results[c][1]:>8.2f}", end="")
        print()

    # ============================================================
    # Part 2: Top 5 combos with upsample 28×28
    # ============================================================
    print("\n" + "=" * 90)
    print("Part 2: Top 5 combos + 28×28 upsample")
    print("=" * 90)

    top5 = sorted_results[:5]
    for combo_name, (_, _, _) in top5:
        block_ids = combos[combo_name]
        cls_results = {}
        for cls_name in classes:
            iauc, pap = evaluate_combo(backbone, block_ids, cls_name, args.data_path, device, upsample_to=28)
            cls_results[cls_name] = (iauc, pap)

        avg_iauc = np.mean([v[0] for v in cls_results.values()])
        avg_pap = np.mean([v[1] for v in cls_results.values()])
        print(f"{combo_name:<30} I-AUC={avg_iauc:>6.2f}%  P-AP={avg_pap:>6.2f}%", end="  |")
        for c in classes:
            print(f" {c[:4]}={cls_results[c][1]:.1f}", end="")
        print()

    print("\nDone!")


if __name__ == '__main__':
    main()
