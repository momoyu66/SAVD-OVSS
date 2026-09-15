"""
utils/metrics.py

This module provides evaluation metrics for semantic segmentation tasks,
specifically Intersection over Union (IoU) and mean IoU (mIoU) calculations.
These metrics are essential for assessing the quality of segmentation predictions
against ground truth labels.

Key Components:
- intersection_over_union: Per-class IoU calculation
- mean_intersection_over_union: Average IoU across all classes
- MeanIoU: Accumulative IoU metric class for batch processing
"""

import torch
import numpy as np

def intersection_over_union(pred, target, num_classes):
    """
    Calculates Intersection over Union (IoU) for each class individually.

    IoU measures the overlap between predicted and ground truth segments for each class.
    For each class c, IoU_c = |pred_c ∩ target_c| / |pred_c ∪ target_c|.

    Args:
        pred (torch.Tensor): Predicted segmentation map, shape `[N, H, W]` or `[H, W]`.
        target (torch.Tensor): Ground truth segmentation map, same shape as pred.
        num_classes (int): Total number of classes (including background).

    Returns:
        List[float]: IoU value for each class (0 to num_classes-1).
                    Returns NaN for classes with no ground truth pixels.
    """
    iou = []
    for cls in range(num_classes):
        # Create binary masks for current class
        pred_inds = pred == cls      # Predicted pixels of class c
        target_inds = target == cls  # Ground truth pixels of class c

        # Calculate intersection (true positives)
        intersection = (pred_inds[target_inds]).long().sum().item()

        # Calculate union (predicted + ground truth - intersection)
        pred_count = pred_inds.long().sum().item()
        target_count = target_inds.long().sum().item()
        union = pred_count + target_count - intersection

        # Handle cases where union is zero (no pixels of this class)
        if union == 0:
            iou.append(float('nan'))  # Return NaN for classes not present in ground truth
        else:
            iou.append(intersection / union)  # IoU = intersection / union

    return iou

def mean_intersection_over_union(pred, target, num_classes):
    """
    Calculates the mean Intersection over Union (mIoU) across all classes.

    mIoU is the primary metric for semantic segmentation evaluation, computed as
    the average of per-class IoU values, ignoring classes not present in ground truth.

    Args:
        pred (torch.Tensor): Predicted segmentation map.
        target (torch.Tensor): Ground truth segmentation map.
        num_classes (int): Total number of classes.

    Returns:
        float: Mean IoU value across all classes (NaN values are ignored).
    """
    iou = intersection_over_union(pred, target, num_classes)
    miou = np.nanmean(iou)  # Average IoU, ignoring NaN values
    return miou

class MeanIoU:
    """
    Accumulative mean IoU metric for batch processing.

    This class efficiently computes mIoU across multiple batches by accumulating
    the confusion matrix (intersection counts) and computing IoU at the end.
    Useful for validation loops where you want to compute metrics over entire datasets.

    Supports different evaluation protocols:
    - BG Include: Include background class (class 0) in mIoU calculation
    - BG Exclude: Exclude background class from mIoU calculation
    """

    def __init__(self, num_classes, device='cpu', ignore_indices=255, include_background=True):
        """
        Initializes the MeanIoU metric tracker.

        Args:
            num_classes (int): Total number of classes in the segmentation task.
            device (str): The device (e.g., "cuda" or "cpu") to store and perform calculations on the confusion matrix. Defaults to 'cpu'.
            ignore_indices (int): Label index to ignore in evaluation (typically 255). Defaults to 255.
            include_background (bool): Whether to include background (class 0) in mIoU calculation.
                                       If False, class 0 is excluded from final mIoU. Defaults to True.
        """
        self.num_classes = num_classes
        self.device = device
        self.ignore_indices = ignore_indices
        if isinstance(ignore_indices, int):
            self.ignore_indices = [ignore_indices]
        self.include_background = include_background
        self.reset()  # Initialize intersection matrix

    def update(self, pred, target):
        """
        Updates the internal confusion matrix with a new batch of predictions.

        Args:
            pred (torch.Tensor): Predicted segmentation maps, shape `[N, H, W]`.
            target (torch.Tensor): Ground truth segmentation maps, shape `[N, H, W]`.
        """
        # Flatten tensors and ensure they are on the same device
        pred = pred.view(-1)
        target = target.view(-1)

        # Remove ignored labels if any (e.g., 255)
        mask = (target >= 0) & (target < self.num_classes)
        for ignore_index in self.ignore_indices:
            mask &= (target != ignore_index)
        pred = pred[mask]
        target = target[mask]
        
        # Accumulate confusion matrix on the device
        intersection_vector = self.num_classes * target.long() + pred.long()
        intersection_counts = torch.bincount(intersection_vector, minlength=self.num_classes**2)
        
        # Move intersection_counts to the same device as self.intersection
        intersection_counts = intersection_counts.to(self.device)
        
        # Accumulate directly on the specified device
        self.intersection += intersection_counts.reshape(self.num_classes, self.num_classes)

    def compute(self):
        """
        Computes the mean IoU from the accumulated confusion matrix.

        Returns:
            float: Mean Intersection over Union across all classes.
        """
        # Extract diagonal (true positives) and row/column sums
        tp = torch.diag(self.intersection)  # True positives for each class
        gt_sum = self.intersection.sum(axis=1)  # Ground truth pixels per class (TP + FN)
        pred_sum = self.intersection.sum(axis=0)  # Predicted pixels per class (TP + FP)

        # IoU = TP / (TP + FP + FN) = TP / (pred_sum + gt_sum - TP)
        iou = tp / (gt_sum + pred_sum - tp)

        # If background should be excluded, skip class 0
        if not self.include_background:
            iou = iou[1:]  # Exclude background class (index 0)

        # Average IoU across classes, ignoring NaN (classes not present in ground truth)
        miou = torch.nanmean(iou).item()
        return miou

    def compute_per_class(self):
        """
        Computes per-class IoU from the accumulated confusion matrix.

        Returns:
            torch.Tensor: IoU for each class.
        """
        tp = torch.diag(self.intersection)
        gt_sum = self.intersection.sum(axis=1)
        pred_sum = self.intersection.sum(axis=0)
        iou = tp / (gt_sum + pred_sum - tp)

        if not self.include_background:
            iou = iou[1:]

        return iou

    def reset(self):
        """
        Resets the internal confusion matrix to zeros.

        Call this at the beginning of each validation epoch.
        """
        self.intersection = torch.zeros((self.num_classes, self.num_classes), dtype=torch.int64, device=self.device)
