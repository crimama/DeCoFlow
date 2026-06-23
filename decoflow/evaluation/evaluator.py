"""
Evaluation Functions for DeCoFlow.

Provides evaluation utilities for continual anomaly detection.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.ndimage import gaussian_filter
from typing import Dict, List, Optional

from decoflow.data.datasets import get_dataset_class


def evaluate_class(trainer,
                   class_name: str,
                   args,
                   task_id: Optional[int] = None,
                   ground_truth_task_id: Optional[int] = None,
                   target_size: int = 224,
                   sigma_override: Optional[float] = None) -> Dict:
    """
    Evaluate a single class.

    Args:
        trainer: DeCoFlowContinualTrainer
        class_name: Class name to evaluate
        args: Configuration
        task_id: If None, use router for task selection
        ground_truth_task_id: Ground truth task ID for routing accuracy calculation
        target_size: Target size for pixel-level evaluation

    Returns:
        Dict containing:
            - img_auc: Image-level AUROC
            - pixel_auc: Pixel-level AUROC
            - img_ap: Image-level Average Precision
            - pixel_ap: Pixel-level Average Precision
            - routing_accuracy: Accuracy of task routing (if using router)
            - routing_distribution: Distribution of predicted tasks
            - n_samples: Number of test samples
    """
    device = trainer.device

    # Get dataset class based on args.dataset
    dataset_name = getattr(args, 'dataset', 'mvtec')
    DatasetClass = get_dataset_class(dataset_name)

    # Create test dataset
    test_dataset = DatasetClass(args.data_path, class_name=class_name, train=False,
                                img_size=args.img_size, crp_size=args.img_size, msk_size=target_size)

    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False,
                            drop_last=False, pin_memory=True, num_workers=4)

    gt_label_list = []
    gt_mask_list = []
    anomaly_scores_list = []
    image_scores_list = []
    predicted_tasks_list = []

    print(f"  Evaluating {class_name}...")

    for idx, (image, label, mask, _, _) in enumerate(test_loader):
        gt_label_list.extend(label.cpu().numpy())
        gt_mask_list.extend(mask.cpu().numpy())

        # Inference
        anomaly_scores, image_scores, predicted_tasks = trainer.inference(
            image, task_id=task_id
        )

        # Resize for pixel-level evaluation
        interp_mode = getattr(args, 'interp_mode', 'bicubic')
        interp_kwargs = {'size': (target_size, target_size), 'mode': interp_mode}
        if interp_mode in ('bilinear', 'bicubic'):
            interp_kwargs['align_corners'] = False
        anomaly_scores_resized = F.interpolate(
            anomaly_scores.unsqueeze(1), **interp_kwargs
        ).squeeze(1)

        anomaly_scores_list.append(anomaly_scores_resized.cpu())
        image_scores_list.append(image_scores.cpu())
        predicted_tasks_list.extend(predicted_tasks.cpu().numpy())

    # Concatenate results
    anomaly_scores_all = torch.cat(anomaly_scores_list, dim=0).numpy()
    image_scores_all = torch.cat(image_scores_list, dim=0).numpy()

    gt_label = np.asarray(gt_label_list, dtype=bool)
    gt_mask = np.squeeze(np.asarray(gt_mask_list, dtype=bool), axis=1)

    # Apply Gaussian smoothing (configurable sigma via --score_smooth_sigma or per-class override)
    smooth_sigma = sigma_override if sigma_override is not None else getattr(args, 'score_smooth_sigma', 4.0)
    if smooth_sigma > 0:
        for i in range(anomaly_scores_all.shape[0]):
            anomaly_scores_all[i] = gaussian_filter(anomaly_scores_all[i], sigma=smooth_sigma)

    # Sanitize scores
    image_scores_all = np.nan_to_num(image_scores_all, nan=0.0, posinf=1.0, neginf=0.0)
    anomaly_scores_all = np.nan_to_num(anomaly_scores_all, nan=0.0, posinf=1.0, neginf=0.0)

    # Compute AUROC
    img_auc = roc_auc_score(gt_label, image_scores_all)
    pixel_auc = roc_auc_score(gt_mask.flatten(), anomaly_scores_all.flatten())

    # Compute Average Precision (AP)
    img_ap = average_precision_score(gt_label, image_scores_all)
    pixel_ap = average_precision_score(gt_mask.flatten(), anomaly_scores_all.flatten())

    # Compute routing statistics
    predicted_tasks_arr = np.array(predicted_tasks_list)
    unique_tasks, counts = np.unique(predicted_tasks_arr, return_counts=True)
    routing_distribution = dict(zip(unique_tasks.tolist(), counts.tolist()))

    is_class_level = bool(getattr(trainer, 'use_class_level', False))
    ground_truth_class_id = None
    class_routing_accuracy = None
    task_routing_accuracy = None
    class_to_task = {}
    if is_class_level:
        all_classes = list(getattr(args, 'task_classes', []))
        if class_name in all_classes:
            ground_truth_class_id = all_classes.index(class_name)
        if getattr(trainer, 'class_router', None) is not None:
            class_to_task = getattr(trainer.class_router, 'class_to_task', {})
        if not class_to_task and hasattr(trainer.nf_model, 'class_to_task'):
            class_to_task = getattr(trainer.nf_model, 'class_to_task', {})

    routing_accuracy = None
    if ground_truth_task_id is not None and task_id is None:
        if is_class_level and ground_truth_class_id is not None:
            correct_class_routing = (predicted_tasks_arr == ground_truth_class_id).sum()
            class_routing_accuracy = correct_class_routing / len(predicted_tasks_arr)
            predicted_task_arr = np.array([
                class_to_task.get(int(class_id), -1) for class_id in predicted_tasks_arr
            ])
            correct_task_routing = (predicted_task_arr == ground_truth_task_id).sum()
            task_routing_accuracy = correct_task_routing / len(predicted_tasks_arr)
            routing_accuracy = class_routing_accuracy
        else:
            correct_routing = (predicted_tasks_arr == ground_truth_task_id).sum()
            routing_accuracy = correct_routing / len(predicted_tasks_arr)
            correct_class_routing = None
            correct_task_routing = None

    # Logging
    if task_id is None:
        print(f"    Routing: {routing_distribution}")
        if is_class_level and class_routing_accuracy is not None:
            print(f"    Class Routing Accuracy: {class_routing_accuracy:.2%} ({int(correct_class_routing)}/{len(predicted_tasks_arr)})")
            print(f"    Task Routing Accuracy: {task_routing_accuracy:.2%} ({int(correct_task_routing)}/{len(predicted_tasks_arr)})")
        elif routing_accuracy is not None:
            print(f"    Routing Accuracy: {routing_accuracy:.2%} ({int(correct_routing)}/{len(predicted_tasks_arr)})")

    return {
        'img_auc': img_auc,
        'pixel_auc': pixel_auc,
        'img_ap': img_ap,
        'pixel_ap': pixel_ap,
        'routing_accuracy': routing_accuracy,
        'class_routing_accuracy': class_routing_accuracy,
        'task_routing_accuracy': task_routing_accuracy,
        'routing_distribution': routing_distribution,
        'n_samples': len(predicted_tasks_arr)
    }


def evaluate_all_tasks(trainer,
                       args,
                       use_router: bool = True,
                       target_size: int = 224,
                       per_class_sigma: Optional[Dict[str, float]] = None) -> Dict:
    """
    Evaluate all learned tasks.

    Args:
        trainer: DeCoFlowContinualTrainer
        args: Configuration
        use_router: If True, use router for task selection
        target_size: Target size for pixel-level evaluation

    Returns:
        results: Dict with class-wise and average metrics including routing info
    """
    results = {
        'classes': [],
        'task_ids': [],
        'img_aucs': [],
        'pixel_aucs': [],
        'img_aps': [],
        'pixel_aps': [],
        'routing_accuracies': [],
        'class_routing_accuracies_list': [],
        'task_routing_accuracies_list': [],
        'routing_distributions': [],
        'n_samples': [],
        'class_img_aucs': {},
        'class_pixel_aucs': {},
        'class_img_aps': {},
        'class_pixel_aps': {},
        'class_routing_accuracies': {},
        'class_task_routing_accuracies': {},
    }

    print("\n" + "="*70)
    print("Evaluating All Tasks" + (" (with Router)" if use_router else " (with Oracle)"))
    print("="*70)

    if not hasattr(trainer, 'task_classes') or not trainer.task_classes:
        print("Warning: No tasks have been trained yet. Returning empty results.")
        results['mean_img_auc'] = 0.0
        results['mean_pixel_auc'] = 0.0
        results['task_avg_img_aucs'] = {}
        results['task_avg_pixel_aucs'] = {}
        return results

    for task_id, task_classes in trainer.task_classes.items():
        print(f"\nTask {task_id}: {task_classes}")

        for class_name in task_classes:
            eval_task_id = None if use_router else task_id

            cls_sigma = per_class_sigma.get(class_name) if per_class_sigma else None
            class_results = evaluate_class(
                trainer, class_name, args,
                task_id=eval_task_id,
                ground_truth_task_id=task_id,
                target_size=target_size,
                sigma_override=cls_sigma
            )

            img_auc = class_results['img_auc']
            pixel_auc = class_results['pixel_auc']
            img_ap = class_results['img_ap']
            pixel_ap = class_results['pixel_ap']
            routing_acc = class_results['routing_accuracy']
            class_routing_acc = class_results.get('class_routing_accuracy')
            task_routing_acc = class_results.get('task_routing_accuracy')

            results['classes'].append(class_name)
            results['task_ids'].append(task_id)
            results['img_aucs'].append(img_auc)
            results['pixel_aucs'].append(pixel_auc)
            results['img_aps'].append(img_ap)
            results['pixel_aps'].append(pixel_ap)
            results['routing_accuracies'].append(routing_acc)
            results['class_routing_accuracies_list'].append(class_routing_acc)
            results['task_routing_accuracies_list'].append(task_routing_acc)
            results['routing_distributions'].append(class_results['routing_distribution'])
            results['n_samples'].append(class_results['n_samples'])

            results['class_img_aucs'][class_name] = img_auc
            results['class_pixel_aucs'][class_name] = pixel_auc
            results['class_img_aps'][class_name] = img_ap
            results['class_pixel_aps'][class_name] = pixel_ap
            results['class_routing_accuracies'][class_name] = routing_acc
            results['class_task_routing_accuracies'][class_name] = task_routing_acc

            if class_routing_acc is not None:
                routing_str = f", Class RA={class_routing_acc:.2%}, Task RA={task_routing_acc:.2%}"
            else:
                routing_str = f", Routing Acc={routing_acc:.2%}" if routing_acc is not None else ""
            print(f"    {class_name}: Image AUC={img_auc:.4f}, Pixel AUC={pixel_auc:.4f}, Image AP={img_ap:.4f}, Pixel AP={pixel_ap:.4f}{routing_str}")

    # Compute averages
    results['mean_img_auc'] = np.mean(results['img_aucs'])
    results['mean_pixel_auc'] = np.mean(results['pixel_aucs'])
    results['mean_img_ap'] = np.mean(results['img_aps'])
    results['mean_pixel_ap'] = np.mean(results['pixel_aps'])

    # Compute overall routing accuracy
    if use_router:
        total_correct = 0
        total_samples = 0
        for i, routing_acc in enumerate(results['routing_accuracies']):
            if routing_acc is not None:
                n = results['n_samples'][i]
                total_correct += routing_acc * n
                total_samples += n
        results['mean_routing_accuracy'] = total_correct / total_samples if total_samples > 0 else None
    else:
        results['mean_routing_accuracy'] = None

    class_routing_values = [
        r for r in results.get('class_routing_accuracies_list', []) if r is not None
    ]
    task_routing_values = [
        r for r in results.get('task_routing_accuracies_list', []) if r is not None
    ]
    results['mean_class_routing_accuracy'] = (
        float(np.mean(class_routing_values)) if class_routing_values else None
    )
    results['mean_task_routing_accuracy'] = (
        float(np.mean(task_routing_values)) if task_routing_values else None
    )

    # Task-wise averages
    results['task_avg_img_aucs'] = {}
    results['task_avg_pixel_aucs'] = {}
    results['task_avg_img_aps'] = {}
    results['task_avg_pixel_aps'] = {}
    results['task_avg_routing_accuracies'] = {}

    for task_id in trainer.task_classes.keys():
        task_indices = [i for i, t in enumerate(results['task_ids']) if t == task_id]
        results['task_avg_img_aucs'][task_id] = np.mean([results['img_aucs'][i] for i in task_indices])
        results['task_avg_pixel_aucs'][task_id] = np.mean([results['pixel_aucs'][i] for i in task_indices])
        results['task_avg_img_aps'][task_id] = np.mean([results['img_aps'][i] for i in task_indices])
        results['task_avg_pixel_aps'][task_id] = np.mean([results['pixel_aps'][i] for i in task_indices])

        task_routing_accs = [results['routing_accuracies'][i] for i in task_indices if results['routing_accuracies'][i] is not None]
        if task_routing_accs:
            results['task_avg_routing_accuracies'][task_id] = np.mean(task_routing_accs)

    # Print summary
    print("\n" + "="*70)
    print("Evaluation Summary")
    print("="*70)

    for i, class_name in enumerate(results['classes']):
        class_ra = results.get('class_routing_accuracies_list', [None] * len(results['classes']))[i]
        task_ra = results.get('task_routing_accuracies_list', [None] * len(results['classes']))[i]
        if class_ra is not None:
            routing_str = f", ClassRA={class_ra:.2%}, TaskRA={task_ra:.2%}"
        else:
            routing_str = f", Routing={results['routing_accuracies'][i]:.2%}" if results['routing_accuracies'][i] is not None else ""
        print(f"  {class_name:15s} (Task {results['task_ids'][i]}): "
              f"I-AUC={results['img_aucs'][i]:.4f}, P-AUC={results['pixel_aucs'][i]:.4f}, "
              f"I-AP={results['img_aps'][i]:.4f}, P-AP={results['pixel_aps'][i]:.4f}{routing_str}")

    print("-"*70)

    for task_id in trainer.task_classes.keys():
        routing_str = ""
        if task_id in results['task_avg_routing_accuracies']:
            routing_str = f", Routing Acc={results['task_avg_routing_accuracies'][task_id]:.2%}"
        print(f"  Task {task_id} Average: "
              f"I-AUC={results['task_avg_img_aucs'][task_id]:.4f}, P-AUC={results['task_avg_pixel_aucs'][task_id]:.4f}, "
              f"I-AP={results['task_avg_img_aps'][task_id]:.4f}, P-AP={results['task_avg_pixel_aps'][task_id]:.4f}{routing_str}")

    print("-"*70)
    if results.get('mean_class_routing_accuracy') is not None:
        routing_str = (
            f", Class RA={results['mean_class_routing_accuracy']:.2%}, "
            f"Task RA={results['mean_task_routing_accuracy']:.2%}"
        )
    else:
        routing_str = f", Routing Acc={results['mean_routing_accuracy']:.2%}" if results['mean_routing_accuracy'] is not None else ""
    print(f"  Overall Average: "
          f"I-AUC={results['mean_img_auc']:.4f}, P-AUC={results['mean_pixel_auc']:.4f}, "
          f"I-AP={results['mean_img_ap']:.4f}, P-AP={results['mean_pixel_ap']:.4f}{routing_str}")
    print("="*70)

    return results


def evaluate_routing_performance(trainer,
                                  args,
                                  target_size: int = 224) -> Dict:
    """
    Detailed evaluation of routing performance.

    Computes:
    1. Overall routing accuracy
    2. Per-task routing accuracy
    3. Confusion matrix of routing
    4. Per-class routing distribution

    Args:
        trainer: DeCoFlowContinualTrainer
        args: Configuration
        target_size: Target size for evaluation

    Returns:
        Dict with detailed routing metrics
    """
    device = trainer.device

    # Get dataset class based on args.dataset
    dataset_name = getattr(args, 'dataset', 'mvtec')
    DatasetClass = get_dataset_class(dataset_name)

    print("\n" + "="*70)
    print("Detailed Routing Performance Evaluation")
    print("="*70)

    routing_results = {
        'class_name': [],
        'ground_truth_task': [],
        'predicted_tasks': [],
        'correct_count': [],
        'total_count': [],
        'accuracy': []
    }

    num_tasks = len(trainer.task_classes)
    confusion_matrix = np.zeros((num_tasks, num_tasks), dtype=int)

    for task_id, task_classes in trainer.task_classes.items():
        print(f"\nTask {task_id}: {task_classes}")

        for class_name in task_classes:
            test_dataset = DatasetClass(args.data_path, class_name=class_name, train=False,
                                        img_size=args.img_size, crp_size=args.img_size, msk_size=target_size)
            test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False,
                                    drop_last=False, pin_memory=True, num_workers=4)

            predicted_tasks_list = []

            trainer.nf_model.eval()
            trainer.vit_extractor.eval()

            with torch.no_grad():
                for images, _, _, _, _ in test_loader:
                    images = images.to(device)
                    image_features = trainer.vit_extractor.get_image_level_features(images)
                    if getattr(trainer, 'use_class_level', False) and getattr(trainer, 'class_router', None) is not None:
                        predicted_classes = trainer.class_router.route(image_features)
                        class_to_task = getattr(trainer.class_router, 'class_to_task', {})
                        predicted_tasks = torch.tensor(
                            [class_to_task.get(int(c), -1) for c in predicted_classes],
                            device=predicted_classes.device,
                            dtype=torch.long,
                        )
                    else:
                        predicted_tasks = trainer.router.route(image_features)
                    predicted_tasks_list.extend(predicted_tasks.cpu().numpy())

            predicted_tasks_arr = np.array(predicted_tasks_list)

            # Compute accuracy
            correct = (predicted_tasks_arr == task_id).sum()
            total = len(predicted_tasks_arr)
            accuracy = correct / total

            routing_results['class_name'].append(class_name)
            routing_results['ground_truth_task'].append(task_id)
            routing_results['predicted_tasks'].append(predicted_tasks_arr)
            routing_results['correct_count'].append(correct)
            routing_results['total_count'].append(total)
            routing_results['accuracy'].append(accuracy)

            # Update confusion matrix
            for pred_task in predicted_tasks_arr:
                if 0 <= pred_task < num_tasks:
                    confusion_matrix[task_id, pred_task] += 1

            # Print per-class routing distribution
            unique_tasks, counts = np.unique(predicted_tasks_arr, return_counts=True)
            dist_str = ", ".join([f"T{t}:{c}" for t, c in zip(unique_tasks, counts)])
            print(f"    {class_name}: Acc={accuracy:.2%} ({correct}/{total}) | Dist: [{dist_str}]")

    # Compute overall statistics
    total_correct = sum(routing_results['correct_count'])
    total_samples = sum(routing_results['total_count'])
    overall_accuracy = total_correct / total_samples

    # Compute per-task statistics
    task_accuracies = {}
    for task_id in trainer.task_classes.keys():
        task_indices = [i for i, t in enumerate(routing_results['ground_truth_task']) if t == task_id]
        task_correct = sum([routing_results['correct_count'][i] for i in task_indices])
        task_total = sum([routing_results['total_count'][i] for i in task_indices])
        task_accuracies[task_id] = task_correct / task_total if task_total > 0 else 0.0

    routing_results['overall_accuracy'] = overall_accuracy
    routing_results['task_accuracies'] = task_accuracies
    routing_results['confusion_matrix'] = confusion_matrix

    # Print summary
    print("\n" + "="*70)
    print("Routing Summary")
    print("="*70)

    print(f"\n  Overall Routing Accuracy: {overall_accuracy:.2%} ({total_correct}/{total_samples})")

    print("\n  Per-Task Routing Accuracy:")
    for task_id, acc in task_accuracies.items():
        classes = trainer.task_classes[task_id]
        print(f"    Task {task_id} ({', '.join(classes)}): {acc:.2%}")

    print("\n  Confusion Matrix (rows=GT, cols=Predicted):")
    task_labels = [f"T{i}" for i in range(num_tasks)]
    print("       " + "  ".join([f"{l:>6}" for l in task_labels]))
    for i, row in enumerate(confusion_matrix):
        row_str = "  ".join([f"{v:>6}" for v in row])
        print(f"    {task_labels[i]}:  {row_str}")

    print("="*70)

    return routing_results
