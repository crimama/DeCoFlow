"""
Training Logger for Continual Learning.

Records all training logs to both console and .log file.
Creates structured logs with timestamps and proper formatting.
Saves evaluation metrics and training logs to CSV files.
"""

import logging
import csv
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

import numpy as np


class TrainingLogger:
    """
    Training Logger for Continual Learning.

    Records all training logs to both console and .log file.
    Creates structured logs with timestamps and proper formatting.
    """
    def __init__(self, log_dir: str = './logs', experiment_name: str = None):
        """
        Initialize training logger.

        Args:
            log_dir: Base directory to save log files
            experiment_name: Name of the experiment (default: timestamp)
        """
        # Create experiment name with timestamp
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"continual_training_{timestamp}"

        self.experiment_name = experiment_name

        # Create experiment-specific directory: log_dir/experiment_name/
        self.log_dir = Path(log_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.log_file = self.log_dir / f"{experiment_name}.log"

        # Setup logger
        self.logger = logging.getLogger(experiment_name)
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()

        # File handler
        file_handler = logging.FileHandler(self.log_file, mode='w')
        file_handler.setLevel(logging.INFO)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        # Track metrics
        self.metrics_history = {
            'task_id': [],
            'epoch': [],
            'stage': [],
            'loss': [],
            'learning_rate': []
        }

        # Track evaluation metrics
        self.evaluation_metrics = {
            'class_name': [],
            'task_id': [],
            'img_auc': [],
            'pixel_auc': [],
            'img_ap': [],
            'pixel_ap': [],
            'routing_accuracy': []
        }

        self.info(f"{'='*70}")
        self.info(f"Training Logger Initialized")
        self.info(f"Log file: {self.log_file}")
        self.info(f"Experiment: {experiment_name}")
        self.info(f"{'='*70}\n")

    def save_config(self, config: Dict[str, Any], filename: str = "config.json"):
        """
        Save experiment configuration to log folder.

        Args:
            config: Dictionary containing all configuration parameters
            filename: Name of the config file (default: config.json)
        """
        config_file = self.log_dir / filename

        # Convert non-serializable objects to strings
        serializable_config = {}
        for key, value in config.items():
            try:
                json.dumps(value)  # Test if serializable
                serializable_config[key] = value
            except (TypeError, ValueError):
                serializable_config[key] = str(value)

        with open(config_file, 'w') as f:
            json.dump(serializable_config, f, indent=4, default=str)

        self.info(f"📝 Config saved to: {config_file}")

    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)

    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)

    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)

    def log_task_start(self, task_id: int, task_classes: List[str], num_epochs: int, lr: float):
        """Log task training start."""
        self.info(f"\n{'='*70}")
        self.info(f"🚀 Training Task {task_id}: {task_classes}")
        self.info(f"{'='*70}")
        self.info(f"   - Epochs: {num_epochs}")
        self.info(f"   - Learning Rate: {lr:.2e}")

    def log_stage_start(self, stage_name: str, num_epochs: int, details: Dict = None):
        """Log training stage start."""
        self.info(f"\n{'─'*70}")
        self.info(f"📌 {stage_name} ({num_epochs} epochs)")
        self.info(f"{'─'*70}")
        if details:
            for key, value in details.items():
                self.info(f"   - {key}: {value}")

    def log_epoch(self, task_id: int, epoch: int, total_epochs: int,
                  avg_loss: float, stage: str = "TRAIN", extra_info: Dict = None):
        """Log epoch results."""
        msg = f"  📊 [{stage}] Epoch [{epoch+1}/{total_epochs}] Average Loss: {avg_loss:.4f}"

        if extra_info:
            for key, value in extra_info.items():
                if isinstance(value, float):
                    msg += f" | {key}: {value:.4f}"
                else:
                    msg += f" | {key}: {value}"

        self.info(msg)

        # Record metrics
        self.metrics_history['task_id'].append(task_id)
        self.metrics_history['epoch'].append(epoch)
        self.metrics_history['stage'].append(stage)
        self.metrics_history['loss'].append(avg_loss)
        self.metrics_history['learning_rate'].append(extra_info.get('LR', 0.0) if extra_info else 0.0)

    def log_batch(self, task_id: int, epoch: int, total_epochs: int,
                  batch_idx: int, total_batches: int, batch_loss: float,
                  avg_loss: float, stage: str = "TRAIN", extra_info: Dict = None):
        """Log batch results."""
        msg = f"  [{stage}] Epoch [{epoch+1}/{total_epochs}] Batch [{batch_idx+1}/{total_batches}] "
        msg += f"Loss: {batch_loss:.4f} | Avg: {avg_loss:.4f}"

        if extra_info:
            for key, value in extra_info.items():
                if isinstance(value, float):
                    msg += f" | {key}: {value:.4f}"
                else:
                    msg += f" | {key}: {value}"

        self.info(msg)

    def log_model_info(self, model_name: str, num_params: int, details: Dict = None):
        """Log model information."""
        self.info(f"\n📦 Model: {model_name}")
        self.info(f"   - Trainable Parameters: {num_params:,}")
        if details:
            for key, value in details.items():
                self.info(f"   - {key}: {value}")

    def log_task_complete(self, task_id: int):
        """Log task completion."""
        self.info(f"\n✅ Task {task_id} training completed!\n")

    def log_prototype_creation(self, task_id: int, n_samples: int, mean_shape: tuple):
        """Log prototype creation."""
        self.info(f"\n📦 Building prototype for Task {task_id}...")
        self.info(f"   ✅ Prototype stored: μ shape={mean_shape}, n_samples={n_samples}")

    def log_feature_stats(self, n_samples: int, mean_norm: float, std_mean: float,
                         std_min: float, std_max: float):
        """Log feature statistics."""
        self.info(f"📊 Feature Statistics Finalized:")
        self.info(f"   - n_samples: {n_samples}")
        self.info(f"   - mean_norm: {mean_norm:.4f}")
        self.info(f"   - std_mean: {std_mean:.4f}")
        self.info(f"   - std_min: {std_min:.4f}, std_max: {std_max:.4f}")

    def log_evaluation(self, task_id: int, metrics: Dict):
        """Log evaluation metrics."""
        self.info(f"\n📊 Evaluation Results for Task {task_id}:")
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, float):
                self.info(f"   - {metric_name}: {metric_value:.4f}")
            else:
                self.info(f"   - {metric_name}: {metric_value}")

    def save_evaluation_metrics(self, results: Dict):
        """Save evaluation metrics to history."""
        # Save class-wise metrics
        for i, class_name in enumerate(results.get('classes', [])):
            self.evaluation_metrics['class_name'].append(class_name)
            self.evaluation_metrics['task_id'].append(results['task_ids'][i])
            self.evaluation_metrics['img_auc'].append(results['img_aucs'][i])
            self.evaluation_metrics['pixel_auc'].append(results['pixel_aucs'][i])
            self.evaluation_metrics['img_ap'].append(results.get('img_aps', [0.0]*len(results['classes']))[i])
            self.evaluation_metrics['pixel_ap'].append(results.get('pixel_aps', [0.0]*len(results['classes']))[i])
            self.evaluation_metrics['routing_accuracy'].append(
                results['routing_accuracies'][i] if results['routing_accuracies'][i] is not None else -1.0
            )

        # Save task-wise averages
        for task_id in results.get('task_avg_img_aucs', {}).keys():
            self.evaluation_metrics['class_name'].append(f'Task_{task_id}_Average')
            self.evaluation_metrics['task_id'].append(task_id)
            self.evaluation_metrics['img_auc'].append(results['task_avg_img_aucs'][task_id])
            self.evaluation_metrics['pixel_auc'].append(results['task_avg_pixel_aucs'][task_id])
            self.evaluation_metrics['img_ap'].append(results.get('task_avg_img_aps', {}).get(task_id, 0.0))
            self.evaluation_metrics['pixel_ap'].append(results.get('task_avg_pixel_aps', {}).get(task_id, 0.0))
            self.evaluation_metrics['routing_accuracy'].append(
                results['task_avg_routing_accuracies'].get(task_id, -1.0) if results.get('task_avg_routing_accuracies') else -1.0
            )

        # Save overall average
        if 'mean_img_auc' in results:
            self.evaluation_metrics['class_name'].append('Overall_Average')
            self.evaluation_metrics['task_id'].append(-1)
            self.evaluation_metrics['img_auc'].append(results['mean_img_auc'])
            self.evaluation_metrics['pixel_auc'].append(results['mean_pixel_auc'])
            self.evaluation_metrics['img_ap'].append(results.get('mean_img_ap', 0.0))
            self.evaluation_metrics['pixel_ap'].append(results.get('mean_pixel_ap', 0.0))
            self.evaluation_metrics['routing_accuracy'].append(
                results.get('mean_routing_accuracy', -1.0) if results.get('mean_routing_accuracy') is not None else -1.0
            )

    def save_training_log_csv(self, task_id: int, epoch: int, iteration: int,
                               losses: Dict[str, Any], lr: float, stage: str = "FAST"):
        """
        Save training iteration logs to CSV file dynamically based on losses dictionary.
        Similar to NFCAD's save_training_log function.

        Args:
            task_id: Current task ID
            epoch: Current epoch
            iteration: Current iteration/batch index
            losses: Dictionary of loss values
            lr: Current learning rate
            stage: Training stage (FAST/SLOW)
        """
        log_file = self.log_dir / f"training_log_task_{task_id}.csv"

        # Helper function to convert tensor to float
        def to_float(value):
            if hasattr(value, 'item'):  # torch.Tensor
                return value.item()
            return float(value) if value is not None else 0.0

        # Check if file exists to write header
        write_header = not log_file.exists()

        # Read existing header if file exists
        existing_header = None
        if not write_header:
            try:
                with open(log_file, 'r', newline='') as f:
                    reader = csv.reader(f)
                    existing_header = next(reader, None)
            except:
                write_header = True

        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)

            if write_header:
                # Dynamic header: base columns + all loss keys (sorted for consistency)
                base_columns = ['timestamp', 'task_id', 'epoch', 'iteration', 'stage']
                loss_keys = sorted([k for k in losses.keys()])
                header = base_columns + loss_keys + ['lr']
                writer.writerow(header)
            else:
                header = existing_header

            # Prepare row data
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            row = [timestamp, task_id, epoch, iteration, stage]

            # Add loss values in the order of header (skip base columns and lr)
            loss_columns = header[5:-1]  # Skip first 5 (base) and last 1 (lr)
            for loss_key in loss_columns:
                row.append(to_float(losses.get(loss_key, 0.0)))

            # Add learning rate at the end
            row.append(lr)

            writer.writerow(row)

    def save_evaluation_results_csv(self, task_id: int, epoch: int,
                                     class_names: List[str],
                                     img_aucs: List[float],
                                     pixel_aucs: List[float],
                                     routing_accuracies: List[Optional[float]] = None):
        """
        Save evaluation results to CSV file.
        Similar to NFCAD's save_evaluation_results function.

        Args:
            task_id: Current task ID (evaluation point)
            epoch: Current epoch
            class_names: List of class names evaluated
            img_aucs: List of image-level AUC scores
            pixel_aucs: List of pixel-level AUC scores
            routing_accuracies: List of routing accuracy scores (optional)
        """
        result_file = self.log_dir / f"evaluation_results_task_{task_id}.csv"

        # Create expected header
        expected_header = ['timestamp', 'task_id', 'epoch']
        for class_name in class_names:
            expected_header.extend([f'{class_name}_img_auc', f'{class_name}_pixel_auc'])
            if routing_accuracies is not None:
                expected_header.append(f'{class_name}_routing_acc')
        expected_header.extend(['mean_img_auc', 'mean_pixel_auc'])
        if routing_accuracies is not None:
            expected_header.append('mean_routing_acc')

        # Check if file exists and validate header compatibility
        write_header = True
        if result_file.exists():
            with open(result_file, 'r', newline='') as f:
                reader = csv.reader(f)
                existing_header = next(reader, None)
                if existing_header == expected_header:
                    write_header = False
                else:
                    self.info(f"Warning: Header mismatch in {result_file}. Recreating file...")

        # Open file in appropriate mode
        file_mode = 'w' if write_header else 'a'

        with open(result_file, file_mode, newline='') as f:
            writer = csv.writer(f)

            if write_header:
                writer.writerow(expected_header)

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            row = [timestamp, task_id, epoch]

            for i, class_name in enumerate(class_names):
                row.extend([img_aucs[i], pixel_aucs[i]])
                if routing_accuracies is not None:
                    row.append(routing_accuracies[i] if routing_accuracies[i] is not None else -1.0)

            row.extend([np.mean(img_aucs), np.mean(pixel_aucs)])
            if routing_accuracies is not None:
                valid_routing = [r for r in routing_accuracies if r is not None]
                row.append(np.mean(valid_routing) if valid_routing else -1.0)

            writer.writerow(row)

    def save_continual_results_csv(self, task_id: int, current_classes: List[str],
                                    all_classes: List[str], img_aucs: List[float],
                                    pixel_aucs: List[float], continual_tasks: List[List[str]],
                                    routing_accuracies: List[Optional[float]] = None):
        """
        Save continual learning results including all previous tasks.
        Similar to NFCAD's save_continual_results function.

        Args:
            task_id: Current task ID
            current_classes: Classes in current task
            all_classes: All classes evaluated
            img_aucs: Image AUC for each class
            pixel_aucs: Pixel AUC for each class
            continual_tasks: List of tasks (list of class lists)
            routing_accuracies: Routing accuracy for each class (optional)
        """
        result_file = self.log_dir / f"continual_results_after_task_{task_id}.csv"

        with open(result_file, 'w', newline='') as f:
            writer = csv.writer(f)

            # Write header
            header = ['class_name', 'task_id', 'img_auc', 'pixel_auc', 'is_current_task']
            if routing_accuracies is not None:
                header.append('routing_accuracy')
            writer.writerow(header)

            # Write results for each class
            for i, class_name in enumerate(all_classes):
                task_idx = -1
                for t_id, task_classes in enumerate(continual_tasks):
                    if class_name in task_classes:
                        task_idx = t_id
                        break

                is_current = class_name in current_classes
                row = [class_name, task_idx, img_aucs[i], pixel_aucs[i], is_current]
                if routing_accuracies is not None:
                    row.append(routing_accuracies[i] if routing_accuracies[i] is not None else -1.0)
                writer.writerow(row)

            # Write summary
            summary_row = ['Average', '', np.mean(img_aucs), np.mean(pixel_aucs), '']
            if routing_accuracies is not None:
                valid_routing = [r for r in routing_accuracies if r is not None]
                summary_row.append(np.mean(valid_routing) if valid_routing else -1.0)
            writer.writerow(summary_row)

            # Write task-wise averages
            for t_id, task_classes in enumerate(continual_tasks[:task_id + 1]):
                task_indices = [i for i, cls in enumerate(all_classes) if cls in task_classes]
                if task_indices:
                    task_img_aucs = [img_aucs[i] for i in task_indices]
                    task_pixel_aucs = [pixel_aucs[i] for i in task_indices]
                    task_row = [f'Task_{t_id}_Average', t_id, np.mean(task_img_aucs), np.mean(task_pixel_aucs), '']
                    if routing_accuracies is not None:
                        task_routing = [routing_accuracies[i] for i in task_indices if routing_accuracies[i] is not None]
                        task_row.append(np.mean(task_routing) if task_routing else -1.0)
                    writer.writerow(task_row)

        self.info(f"Continual results saved to: {result_file}")

    def save_unified_evaluation_csv(self, task_id: int, epoch: int,
                                     all_classes: List[str], img_aucs: List[float],
                                     pixel_aucs: List[float], ALL_CLASSES: List[str],
                                     routing_accuracies: List[Optional[float]] = None):
        """
        Save evaluation results for all tasks in a unified file.
        Similar to NFCAD's save_unified_evaluation_results function.

        Args:
            task_id: Current task ID (evaluation point)
            epoch: Current epoch
            all_classes: Classes evaluated at this point
            img_aucs: Image AUC for each evaluated class
            pixel_aucs: Pixel AUC for each evaluated class
            ALL_CLASSES: All classes in the entire experiment
            routing_accuracies: Routing accuracy for each class (optional)
        """
        result_file = self.log_dir / "unified_evaluation_results.csv"

        # Check if file exists to write header
        write_header = not result_file.exists()

        with open(result_file, 'a', newline='') as f:
            writer = csv.writer(f)

            if write_header:
                # Create header with all possible classes
                header = ['timestamp', 'current_task_id', 'epoch']
                for class_name in ALL_CLASSES:
                    header.extend([f'{class_name}_img_auc', f'{class_name}_pixel_auc'])
                    if routing_accuracies is not None:
                        header.append(f'{class_name}_routing_acc')
                header.extend(['mean_img_auc', 'mean_pixel_auc'])
                if routing_accuracies is not None:
                    header.append('mean_routing_acc')
                writer.writerow(header)

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            row = [timestamp, task_id, epoch]

            # Create mappings from class name to metrics
            class_to_img_auc = {}
            class_to_pixel_auc = {}
            class_to_routing = {}

            for i, class_name in enumerate(all_classes):
                class_to_img_auc[class_name] = img_aucs[i]
                class_to_pixel_auc[class_name] = pixel_aucs[i]
                if routing_accuracies is not None:
                    class_to_routing[class_name] = routing_accuracies[i]

            # Add metrics for all classes (use -1 for not evaluated classes)
            for class_name in ALL_CLASSES:
                if class_name in class_to_img_auc:
                    row.extend([class_to_img_auc[class_name], class_to_pixel_auc[class_name]])
                    if routing_accuracies is not None:
                        r = class_to_routing.get(class_name)
                        row.append(r if r is not None else -1.0)
                else:
                    row.extend([-1, -1])  # Not evaluated yet
                    if routing_accuracies is not None:
                        row.append(-1.0)

            row.extend([np.mean(img_aucs), np.mean(pixel_aucs)])
            if routing_accuracies is not None:
                valid_routing = [r for r in routing_accuracies if r is not None]
                row.append(np.mean(valid_routing) if valid_routing else -1.0)

            writer.writerow(row)

    def save_final_results_table(self, results: Dict, continual_tasks: List[List[str]]):
        """
        Save final results in a simple table format.

        Output format:
        Task ID,Class Name,Routing Acc (%),Image AUC,Pixel AUC,Image AP,Pixel AP
        0,leather,100.00,1.0000,0.9481,0.9500,0.4200
        ...
        Mean,Overall,98.43,0.7513*,0.8464*,0.8500*,0.4000*

        Args:
            results: Dictionary from evaluate_all_tasks containing:
                - classes: List of class names
                - task_ids: List of task IDs
                - img_aucs: List of image AUCs
                - pixel_aucs: List of pixel AUCs
                - img_aps: List of image APs
                - pixel_aps: List of pixel APs
                - routing_accuracies: List of routing accuracies (can be None)
            continual_tasks: List of task class lists for task ID mapping
        """
        result_file = self.log_dir / "final_results.csv"

        classes = results.get('classes', [])
        task_ids = results.get('task_ids', [])
        img_aucs = results.get('img_aucs', [])
        pixel_aucs = results.get('pixel_aucs', [])
        img_aps = results.get('img_aps', [])
        pixel_aps = results.get('pixel_aps', [])
        routing_accuracies = results.get('routing_accuracies', [])

        with open(result_file, 'w', newline='') as f:
            writer = csv.writer(f)

            # Write header
            writer.writerow(['Task ID', 'Class Name', 'Routing Acc (%)', 'Image AUC', 'Pixel AUC', 'Image AP', 'Pixel AP'])

            # Write per-class results
            for i, class_name in enumerate(classes):
                task_id = task_ids[i]
                img_auc = img_aucs[i]
                pixel_auc = pixel_aucs[i]
                img_ap = img_aps[i] if i < len(img_aps) else 0.0
                pixel_ap = pixel_aps[i] if i < len(pixel_aps) else 0.0

                # Format routing accuracy as percentage
                routing_acc = routing_accuracies[i] if i < len(routing_accuracies) else None
                if routing_acc is not None:
                    routing_acc_str = f"{routing_acc * 100:.2f}"
                else:
                    routing_acc_str = "N/A"

                writer.writerow([
                    task_id,
                    class_name,
                    routing_acc_str,
                    f"{img_auc:.4f}",
                    f"{pixel_auc:.4f}",
                    f"{img_ap:.4f}",
                    f"{pixel_ap:.4f}"
                ])

            # Write mean row
            mean_img_auc = np.mean(img_aucs) if img_aucs else 0.0
            mean_pixel_auc = np.mean(pixel_aucs) if pixel_aucs else 0.0
            mean_img_ap = np.mean(img_aps) if img_aps else 0.0
            mean_pixel_ap = np.mean(pixel_aps) if pixel_aps else 0.0

            # Calculate mean routing accuracy
            valid_routing = [r for r in routing_accuracies if r is not None]
            if valid_routing:
                mean_routing_acc = np.mean(valid_routing) * 100
                mean_routing_str = f"{mean_routing_acc:.2f}"
            else:
                mean_routing_str = "N/A"

            writer.writerow([
                'Mean',
                'Overall',
                mean_routing_str,
                f"{mean_img_auc:.4f}*",
                f"{mean_pixel_auc:.4f}*",
                f"{mean_img_ap:.4f}*",
                f"{mean_pixel_ap:.4f}*"
            ])

        self.info(f"\n📊 Final results table saved to: {result_file}")

        # Also print the table to console/log
        self.info("\n" + "="*90)
        self.info("Final Results Table")
        self.info("="*90)
        self.info(f"{'Task ID':<10}{'Class Name':<15}{'Routing (%)':<14}{'I-AUC':<10}{'P-AUC':<10}{'I-AP':<10}{'P-AP':<10}")
        self.info("-"*90)

        for i, class_name in enumerate(classes):
            task_id = task_ids[i]
            img_auc = img_aucs[i]
            pixel_auc = pixel_aucs[i]
            img_ap = img_aps[i] if i < len(img_aps) else 0.0
            pixel_ap = pixel_aps[i] if i < len(pixel_aps) else 0.0
            routing_acc = routing_accuracies[i] if i < len(routing_accuracies) else None

            if routing_acc is not None:
                routing_acc_str = f"{routing_acc * 100:.2f}"
            else:
                routing_acc_str = "N/A"

            self.info(f"{task_id:<10}{class_name:<15}{routing_acc_str:<14}{img_auc:<10.4f}{pixel_auc:<10.4f}{img_ap:<10.4f}{pixel_ap:<10.4f}")

        self.info("-"*90)
        self.info(f"{'Mean':<10}{'Overall':<15}{mean_routing_str:<14}{mean_img_auc:<10.4f}{mean_pixel_auc:<10.4f}{mean_img_ap:<10.4f}{mean_pixel_ap:<10.4f}")
        self.info("="*90)

    def save_final_summary(self, strategy_name: str, all_classes: List[str],
                           img_aucs: List[float], pixel_aucs: List[float],
                           num_tasks: int, ablation_config: Any = None,
                           routing_accuracy: Optional[float] = None,
                           additional_metrics: Dict = None,
                           img_aps: List[float] = None,
                           pixel_aps: List[float] = None):
        """
        Save final summary with experiment configuration.
        Similar to NFCAD's save_final_summary function.

        Args:
            strategy_name: Name of the training strategy
            all_classes: All classes in the experiment
            img_aucs: Final image AUC for each class
            pixel_aucs: Final pixel AUC for each class
            num_tasks: Total number of tasks
            ablation_config: Ablation configuration object (optional)
            routing_accuracy: Overall routing accuracy (optional)
            additional_metrics: Additional metrics to save (optional)
            img_aps: Final image AP for each class (optional)
            pixel_aps: Final pixel AP for each class (optional)
        """
        final_summary_file = self.log_dir / f"final_summary_{self.experiment_name}.csv"

        with open(final_summary_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['metric', 'value'])
            writer.writerow(['experiment_name', self.experiment_name])
            writer.writerow(['strategy', strategy_name])
            writer.writerow(['final_average_img_auc', np.mean(img_aucs)])
            writer.writerow(['final_average_pixel_auc', np.mean(pixel_aucs)])
            if img_aps:
                writer.writerow(['final_average_img_ap', np.mean(img_aps)])
            if pixel_aps:
                writer.writerow(['final_average_pixel_ap', np.mean(pixel_aps)])
            writer.writerow(['total_classes', len(all_classes)])
            writer.writerow(['total_tasks', num_tasks])

            if routing_accuracy is not None:
                writer.writerow(['final_routing_accuracy', routing_accuracy])

            # Add ablation configuration
            if ablation_config is not None:
                writer.writerow(['---ablation_config---', ''])

                tracked_fields = [
                    # Core toggles
                    'use_lora', 'use_router', 'use_task_adapter',
                    'use_pos_embedding', 'use_task_bias', 'use_mahalanobis',
                    # Critical score/tail settings
                    'score_aggregation_mode', 'score_aggregation_percentile',
                    'score_aggregation_top_k', 'score_aggregation_top_k_percent',
                    'use_tail_aware_loss', 'tail_weight', 'tail_top_k_ratio',
                    'tail_match_eval_topk', 'tail_loss_blend_mode',
                    # ACB detail needed for post-hoc attribution
                    'acb_n_blocks', 'acb_subnet_type', 'acb_kernel_size',
                    # Related regularization knobs
                    'lambda_logdet', 'focal_gamma', 'score_smooth_sigma',
                ]

                for field in tracked_fields:
                    if hasattr(ablation_config, field):
                        writer.writerow([field, getattr(ablation_config, field)])

            # Add class-wise results
            writer.writerow(['---class_results---', ''])
            for i, class_name in enumerate(all_classes):
                writer.writerow([f'{class_name}_img_auc', img_aucs[i]])
                writer.writerow([f'{class_name}_pixel_auc', pixel_aucs[i]])
                if img_aps and i < len(img_aps):
                    writer.writerow([f'{class_name}_img_ap', img_aps[i]])
                if pixel_aps and i < len(pixel_aps):
                    writer.writerow([f'{class_name}_pixel_ap', pixel_aps[i]])

            # Add additional metrics
            if additional_metrics:
                writer.writerow(['---additional_metrics---', ''])
                for metric, value in additional_metrics.items():
                    writer.writerow([metric, value])

        self.info(f"\n📊 Final summary saved to: {final_summary_file}")
        self.info(f"   Final Average Image AUC: {np.mean(img_aucs):.4f}")
        self.info(f"   Final Average Pixel AUC: {np.mean(pixel_aucs):.4f}")
        if img_aps:
            self.info(f"   Final Average Image AP: {np.mean(img_aps):.4f}")
        if pixel_aps:
            self.info(f"   Final Average Pixel AP: {np.mean(pixel_aps):.4f}")
        if routing_accuracy is not None:
            self.info(f"   Final Routing Accuracy: {routing_accuracy:.4f}")

    def save_metrics_summary(self):
        """Save metrics summary to CSV."""
        import pandas as pd

        csv_file = self.log_dir / f"{self.experiment_name}_metrics.csv"

        # Combine training and evaluation metrics
        all_data = []

        # Add training metrics
        if self.metrics_history['task_id']:
            df_train = pd.DataFrame(self.metrics_history)
            all_data.append(df_train)

        # Add evaluation metrics if available
        if self.evaluation_metrics['class_name']:
            df_eval = pd.DataFrame(self.evaluation_metrics)
            all_data.append(df_eval)

        # Save combined metrics
        if all_data:
            df_combined = pd.concat(all_data, ignore_index=True)
            df_combined.to_csv(csv_file, index=False)
            self.info(f"\n📊 Metrics summary saved to: {csv_file}")
            self.info(f"   - Training metrics: {len(self.metrics_history['task_id'])} rows")
            self.info(f"   - Evaluation metrics: {len(self.evaluation_metrics['class_name'])} rows")

    def close(self):
        """Close logger and save final summary."""
        self.save_metrics_summary()
        self.info(f"\n{'='*70}")
        self.info(f"Training Completed")
        self.info(f"{'='*70}")

        # Close handlers
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)


def setup_training_logger(log_dir: str = './logs', experiment_name: str = None) -> TrainingLogger:
    """
    Setup training logger.

    Args:
        log_dir: Directory to save log files
        experiment_name: Name of the experiment

    Returns:
        TrainingLogger instance
    """
    return TrainingLogger(log_dir=log_dir, experiment_name=experiment_name)
