"""
ViT Block-level Feature Analysis for VisA
==========================================
분석 목표:
1. 각 블록(0~11)의 feature 특성: 분산, entropy, spatial structure
2. 블록별 단독 anomaly detection 성능 (I-AUC, P-AP)
3. Shallow(0-3) vs Mid(4-7) vs Deep(8-11) feature의 역할 차이
4. Feature upsample 시 각 블록의 정보 보존 정도

사용법:
    python scripts/analyze_vit_blocks.py --gpu 0
"""

import sys
sys.path.insert(0, '/Volume/DeCoFlow')

import argparse
import torch
import torch.nn.functional as F
import numpy as np
import timm
import csv
import os
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score, average_precision_score

from decoflow.data.visa import VISA as VisADataset


def extract_block_features(backbone, images, block_indices, device):
    """Extract features from specific ViT blocks."""
    backbone.eval()
    images = images.to(device)

    # Hook to capture intermediate features
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


def features_to_spatial(feat, remove_cls=True):
    """Convert (B, N+1, D) to (B, H, W, D)."""
    if remove_cls:
        feat = feat[:, 1:, :]  # Remove CLS token
    B, N, D = feat.shape
    H = W = int(np.sqrt(N))
    return feat.reshape(B, H, W, D)


def compute_simple_anomaly_score(features_spatial, train_mean, train_std):
    """Simple Mahalanobis-like anomaly score per patch."""
    # Normalize
    z = (features_spatial - train_mean) / (train_std + 1e-8)
    # Score = sum of squared z-scores per patch
    scores = (z ** 2).sum(dim=-1)  # (B, H, W)
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--data_path', default='/Volume/VisA')
    parser.add_argument('--backbone', default='vit_base_patch16_224.augreg2_in21k_ft_in1k')
    parser.add_argument('--test_classes', nargs='+',
                        default=['cashew', 'pcb1', 'macaroni1', 'capsules', 'chewinggum', 'pipe_fryum'])
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}')

    # Load backbone
    print(f"Loading backbone: {args.backbone}")
    backbone = timm.create_model(args.backbone, pretrained=True, img_size=224)
    backbone = backbone.to(device)
    backbone.eval()

    all_blocks = list(range(12))

    # ============================================================
    # Analysis 1: Feature Statistics per Block
    # ============================================================
    print("\n" + "="*70)
    print("Analysis 1: Feature Statistics per Block (using cashew train set)")
    print("="*70)

    # Use one class for feature analysis
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = VisADataset(
        root=args.data_path,
        class_name='cashew',
        train=True,
        img_size=224,
        crp_size=224,
        msk_size=256
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

    # Collect features from all blocks for first batch
    images, _, _, _, _ = next(iter(loader))
    features = extract_block_features(backbone, images, all_blocks, device)

    print(f"\n{'Block':>6} {'Mean':>10} {'Std':>10} {'Var':>10} {'L2_norm':>10} {'Entropy_est':>12} {'Spatial_var':>12}")
    print("-"*78)

    block_stats = {}
    for idx in all_blocks:
        feat = features_to_spatial(features[idx])  # (B, 14, 14, 768)
        feat_np = feat.cpu().numpy()

        mean_val = feat_np.mean()
        std_val = feat_np.std()
        var_val = feat_np.var()
        l2_norm = np.sqrt((feat_np**2).sum(axis=-1)).mean()

        # Spatial variance: how much does the feature vary across spatial positions?
        # High = position-dependent (good for localization), Low = uniform
        spatial_mean = feat_np.mean(axis=(0,))  # (14, 14, 768) mean across batch
        spatial_var = spatial_mean.var(axis=(0, 1)).mean()  # variance across spatial positions

        # Entropy estimate (via singular value spectrum of patch features)
        feat_flat = feat_np.reshape(-1, 768)  # (B*196, 768)
        if feat_flat.shape[0] > 1000:
            feat_flat = feat_flat[np.random.choice(feat_flat.shape[0], 1000, replace=False)]
        _, S, _ = np.linalg.svd(feat_flat - feat_flat.mean(axis=0), full_matrices=False)
        S_norm = S / S.sum()
        entropy = -np.sum(S_norm * np.log(S_norm + 1e-10))

        block_stats[idx] = {
            'mean': mean_val, 'std': std_val, 'var': var_val,
            'l2_norm': l2_norm, 'entropy': entropy, 'spatial_var': spatial_var
        }

        print(f"{idx:>6} {mean_val:>10.4f} {std_val:>10.4f} {var_val:>10.4f} "
              f"{l2_norm:>10.2f} {entropy:>12.2f} {spatial_var:>12.6f}")

    # ============================================================
    # Analysis 2: Per-block Anomaly Detection Performance
    # ============================================================
    print("\n" + "="*70)
    print("Analysis 2: Per-block Anomaly Detection (single block, simple scoring)")
    print("="*70)

    results = {}

    for cls_name in args.test_classes:
        print(f"\n--- {cls_name} ---")

        # Train set: compute mean/std per block
        train_dataset = VisADataset(
            root=args.data_path,
            class_name=cls_name,
            train=True,
            img_size=224,
            crp_size=224,
            msk_size=256
        )
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=False)

        # Collect train features per block
        train_features = {idx: [] for idx in all_blocks}
        for images, _, _, _, _ in train_loader:
            feats = extract_block_features(backbone, images, all_blocks, device)
            for idx in all_blocks:
                spatial = features_to_spatial(feats[idx])
                train_features[idx].append(spatial.cpu())

        train_stats = {}
        for idx in all_blocks:
            all_feat = torch.cat(train_features[idx], dim=0)  # (N, 14, 14, 768)
            train_stats[idx] = {
                'mean': all_feat.mean(dim=0, keepdim=True),  # (1, 14, 14, 768)
                'std': all_feat.std(dim=0, keepdim=True) + 1e-8
            }

        # Test set
        test_dataset = VisADataset(
            root=args.data_path,
            class_name=cls_name,
            train=False,
            img_size=224,
            crp_size=224,
            msk_size=256
        )
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

        # Evaluate each block
        block_results = {}
        for idx in all_blocks:
            all_scores = []
            all_img_scores = []
            all_labels = []
            all_masks = []

            for images, labels, masks, _, _ in test_loader:
                feats = extract_block_features(backbone, images, [idx], device)
                spatial = features_to_spatial(feats[idx])  # (B, 14, 14, 768)

                # Anomaly score
                scores = compute_simple_anomaly_score(
                    spatial.cpu(), train_stats[idx]['mean'], train_stats[idx]['std']
                )  # (B, 14, 14)

                # Upsample to mask size
                scores_up = F.interpolate(
                    scores.unsqueeze(1).float(), size=(256, 256), mode='bicubic', align_corners=False
                ).squeeze(1)

                # Image score = max of spatial scores
                img_scores = scores.reshape(scores.shape[0], -1).max(dim=1)[0]

                all_scores.append(scores_up.numpy())
                all_img_scores.append(img_scores.numpy())
                all_labels.extend(labels.numpy())
                all_masks.extend(masks.numpy())

            all_scores = np.concatenate(all_scores, axis=0)
            all_img_scores = np.concatenate(all_img_scores, axis=0)
            all_labels = np.array(all_labels, dtype=bool)
            all_masks = np.squeeze(np.array(all_masks, dtype=bool), axis=1)

            # Smooth
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

            block_results[idx] = {'img_auc': img_auc, 'pixel_ap': pixel_ap}

        results[cls_name] = block_results

        # Print
        print(f"{'Block':>6} {'I-AUC%':>8} {'P-AP%':>8}")
        print("-"*26)
        for idx in all_blocks:
            r = block_results[idx]
            print(f"{idx:>6} {r['img_auc']:>8.2f} {r['pixel_ap']:>8.2f}")

    # ============================================================
    # Analysis 3: Block Combination Analysis
    # ============================================================
    print("\n" + "="*70)
    print("Analysis 3: Block Combinations (mean aggregation)")
    print("="*70)

    combos = {
        'shallow [0,1,2]': [0, 1, 2],
        'mid [4,5,6]': [4, 5, 6],
        'deep [9,10,11]': [9, 10, 11],
        'current [3,6,9,11]': [3, 6, 9, 11],
        'all_even [0,2,4,6,8,10]': [0, 2, 4, 6, 8, 10],
        'deep4 [8,9,10,11]': [8, 9, 10, 11],
    }

    cls_name = 'cashew'  # Representative class
    print(f"\nUsing class: {cls_name}")

    train_dataset = VisADataset(
        root=args.data_path, class_name=cls_name, train=True,
        img_size=224, crp_size=224, msk_size=256
    )
    test_dataset = VisADataset(
        root=args.data_path, class_name=cls_name, train=False,
        img_size=224, crp_size=224, msk_size=256
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    for combo_name, block_ids in combos.items():
        # Train stats for combined features
        combo_train_feats = []
        for images, _, _, _, _ in train_loader:
            feats = extract_block_features(backbone, images, block_ids, device)
            combined = torch.stack([features_to_spatial(feats[b]) for b in block_ids], dim=0).mean(dim=0)
            combo_train_feats.append(combined.cpu())

        combo_train = torch.cat(combo_train_feats, dim=0)
        combo_mean = combo_train.mean(dim=0, keepdim=True)
        combo_std = combo_train.std(dim=0, keepdim=True) + 1e-8

        # Test
        all_scores, all_img_scores, all_labels, all_masks = [], [], [], []
        for images, labels, masks, _, _ in test_loader:
            feats = extract_block_features(backbone, images, block_ids, device)
            combined = torch.stack([features_to_spatial(feats[b]) for b in block_ids], dim=0).mean(dim=0)

            scores = compute_simple_anomaly_score(combined.cpu(), combo_mean, combo_std)
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

        print(f"{combo_name:<28} I-AUC={img_auc:>6.2f}%  P-AP={pixel_ap:>6.2f}%")

    # ============================================================
    # Analysis 4: Feature Upsample Effect
    # ============================================================
    print("\n" + "="*70)
    print("Analysis 4: Feature Upsample Effect (14→28 vs 14→56)")
    print("="*70)

    block_ids = [3, 6, 9, 11]  # Current best combo

    for target_size in [14, 28, 56]:
        # Train
        combo_train_feats = []
        for images, _, _, _, _ in train_loader:
            feats = extract_block_features(backbone, images, block_ids, device)
            combined = torch.stack([features_to_spatial(feats[b]) for b in block_ids], dim=0).mean(dim=0)
            # (B, 14, 14, 768)

            if target_size != 14:
                B_t = combined.shape[0]
                combined = combined.permute(0, 3, 1, 2)  # (B, 768, 14, 14)
                combined = F.interpolate(combined, size=(target_size, target_size), mode='bilinear', align_corners=False)
                combined = combined.permute(0, 2, 3, 1)  # (B, H, W, 768)

            combo_train_feats.append(combined.cpu())

        combo_train = torch.cat(combo_train_feats, dim=0)
        combo_mean = combo_train.mean(dim=0, keepdim=True)
        combo_std = combo_train.std(dim=0, keepdim=True) + 1e-8

        # Test
        all_scores, all_img_scores, all_labels, all_masks = [], [], [], []
        for images, labels, masks, _, _ in test_loader:
            feats = extract_block_features(backbone, images, block_ids, device)
            combined = torch.stack([features_to_spatial(feats[b]) for b in block_ids], dim=0).mean(dim=0)

            if target_size != 14:
                B_t = combined.shape[0]
                combined = combined.permute(0, 3, 1, 2)
                combined = F.interpolate(combined, size=(target_size, target_size), mode='bilinear', align_corners=False)
                combined = combined.permute(0, 2, 3, 1)

            scores = compute_simple_anomaly_score(combined.cpu(), combo_mean, combo_std)
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

        print(f"  {target_size}×{target_size} → 256×256  I-AUC={img_auc:>6.2f}%  P-AP={pixel_ap:>6.2f}%")

    print("\n" + "="*70)
    print("Analysis Complete!")
    print("="*70)


if __name__ == '__main__':
    main()
