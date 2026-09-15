"""
utils/model_utils.py

This module provides utility functions for analyzing PyTorch models,
particularly for counting parameters and understanding model complexity.

Key Components:
- count_parameters: Function to count total and trainable parameters in a model
"""

def count_parameters(model):
    """
    Counts the total number of parameters and trainable parameters in a PyTorch model.

    This function iterates through all parameters in the model and separates them
    into total parameters (all parameters) and trainable parameters (parameters
    that require gradients and can be updated during training).

    Args:
        model (torch.nn.Module): The PyTorch model to analyze.

    Returns:
        Dict[str, int]: Dictionary containing:
            - "total_params": Total number of parameters in the model
            - "trainable_params": Number of parameters that require gradients
    """
    total_params = 0
    trainable_params = 0

    # Iterate through all parameters in the model
    for param in model.parameters():
        total_params += param.numel()  # Count all parameters

        if param.requires_grad:
            trainable_params += param.numel()  # Count only trainable parameters

    return {
        "total_params": total_params,
        "trainable_params": trainable_params
    }
