"""
Helper utilities for DeCoFlow.

Contains seed initialization and learning rate parameter settings.
"""

import math
import random
import numpy as np
import torch


def init_seeds(seed: int = 0):
    """
    Initialize random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setting_lr_parameters(args):
    """
    Set learning rate schedule parameters.

    Args:
        args: Configuration object with lr_decay_epochs, meta_epochs, lr_warm,
              lr_cosine, lr_decay_rate, lr_warm_epochs attributes
    """
    args.scaled_lr_decay_epochs = [i * args.meta_epochs // 100 for i in args.lr_decay_epochs]
    print('LR schedule: {}'.format(args.scaled_lr_decay_epochs))

    if args.lr_warm:
        args.lr_warmup_from = args.lr / 10.0
        if args.lr_cosine:
            eta_min = args.lr * (args.lr_decay_rate ** 3)
            args.lr_warmup_to = eta_min + (args.lr - eta_min) * (
                    1 + math.cos(math.pi * args.lr_warm_epochs / args.meta_epochs)) / 2
        else:
            args.lr_warmup_to = args.lr
