"""
model/components.py

Core building blocks shared by the DINOde open-vocabulary semantic segmentation model:
the two frozen encoders and the tensor / prompt utilities they rely on.

Key Components:
- l2norm, l2norm_hw: L2 normalization for vector and spatial-map tensors
- build_prompts: per-protocol prompt template expansion for a class name
- CLIPTextEncoder: frozen CLIP text encoder (open_clip) for text feature extraction
- DinoV3HFBackbone: Hugging Face DINOv3 wrapper for dense patch-grid feature extraction

The flow networks that map between the two feature spaces live in `model/dinode.py`.
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict

import os
import math
import json
import logging
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional dependencies - used for image processing and model loading
try:
    from PIL import Image
    _HAS_PIL = True  # Flag to track PIL availability
except Exception:
    _HAS_PIL = False

try:
    from transformers import AutoImageProcessor, AutoModel
    _HAS_HF = True  # Flag to track Hugging Face transformers availability
except Exception:
    _HAS_HF = False

# -------------------------
# Logging Configuration
# -------------------------
# Configure logging for the components module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("OVSS-DINOv3-COMPONENTS")


# -------------------------
# Utility Functions for Tensor Operations and Visualization
# -------------------------

def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """
    Applies L2 normalization to a tensor along a specified dimension.

    L2 normalization scales the tensor so that its norm along the specified dimension
    equals 1, which is crucial for cosine similarity calculations in contrastive learning.

    Args:
        x (torch.Tensor): The input tensor to be normalized.
        dim (int): The dimension along which to compute the L2 norm. Defaults to -1 (last dimension).
        eps (float): A small epsilon value added to the norm to prevent division by zero. Defaults to 1e-6.

    Returns:
        torch.Tensor: The L2-normalized tensor with the same shape as input.
    """
    norm = x.norm(p=2, dim=dim, keepdim=True)  # Compute L2 norm along specified dimension
    return x / (norm + eps)  # Normalize and add epsilon for numerical stability

def l2norm_hw(z: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Applies per-pixel L2 normalization to a feature map across the channel dimension.

    This function normalizes each spatial location (pixel) independently by computing
    the L2 norm across the channel dimension and scaling the features accordingly.
    This is particularly useful for feature maps where each spatial position represents
    a feature vector that should be normalized.

    Args:
        z (torch.Tensor): The input feature map tensor, shape `[B, C, H, W]` where
                         B=batch size, C=channels, H=height, W=width.
        eps (float): A small value added to the norm to prevent division by zero. Defaults to 1e-6.

    Returns:
        torch.Tensor: The per-pixel L2-normalized feature map with same shape as input.
    """
    # Compute the L2 norm for each pixel across channels (dim=1)
    # Then take square root and add epsilon for numerical stability
    norm_per_pixel = z.pow(2).sum(dim=1, keepdim=True).add(eps).sqrt()
    # Normalize the feature map by dividing by the per-pixel norm
    return z / norm_per_pixel

# Default templates for English language prompt generation
# These templates are used to create varied text descriptions for visual concepts
DEFAULT_TEMPLATES_EN = [
    "a photo of a {}",
    "a cropped photo of a {}",
    "a {}",
    "the {}",
    "a photo of the {}",
]
OPENAI_TEMPLATES_EN = [     # == FULL IMAGENET TEMPLATES
    "a bad photo of a {}.",
    "a photo of many {}.",
    "a sculpture of a {}.",
    "a photo of the hard to see {}.",
    "a low resolution photo of the {}.",
    "a rendering of a {}.",
    "graffiti of a {}.",
    "a bad photo of the {}.",
    "a cropped photo of the {}.",
    "a tattoo of a {}.",
    "the embroidered {}.",
    "a photo of a hard to see {}.",
    "a bright photo of a {}.",
    "a photo of a clean {}.",
    "a photo of a dirty {}.",
    "a dark photo of the {}.",
    "a drawing of a {}.",
    "a photo of my {}.",
    "the plastic {}.",
    "a photo of the cool {}.",
    "a close-up photo of a {}.",
    "a black and white photo of the {}.",
    "a painting of the {}.",
    "a painting of a {}.",
    "a pixelated photo of the {}.",
    "a sculpture of the {}.",
    "a bright photo of the {}.",
    "a cropped photo of a {}.",
    "a plastic {}.",
    "a photo of the dirty {}.",
    "a jpeg corrupted photo of a {}.",
    "a blurry photo of the {}.",
    "a photo of the {}.",
    "a good photo of the {}.",
    "a rendering of the {}.",
    "a {} in a video game.",
    "a photo of one {}.",
    "a doodle of a {}.",
    "a close-up photo of the {}.",
    "a photo of a {}.",
    "the origami {}.",
    "the {} in a video game.",
    "a sketch of a {}.",
    "a doodle of the {}.",
    "a origami {}.",
    "a low resolution photo of a {}.",
    "the toy {}.",
    "a rendition of the {}.",
    "a photo of the clean {}.",
    "a photo of a large {}.",
    "a photo of a nice {}.",
    "a photo of a weird {}.",
    "a blurry photo of a {}.",
    "a cartoon {}.",
    "art of a {}.",
    "a sketch of the {}.",
    "a embroidered {}.",
    "a pixelated photo of a {}.",
    "itap of the {}.",
    "a jpeg corrupted photo of the {}.",
    "a good photo of the {}.",
    "a plushie {}.",
    "a photo of the nice {}.",
    "a photo of the small {}.",
    "a photo of the weird {}.",
    "the cartoon {}.",
    "art of the {}.",
    "a drawing of the {}.",
    "a photo of the large {}.",
    "a black and white photo of a {}.",
    "the plushie {}.",
    "a dark photo of a {}.",
    "itap of a {}.",
    "graffiti of the {}.",
    "a toy {}.",
    "itap of my {}.",
    "a photo of a cool {}.",
    "a photo of a small {}.",
    "a tattoo of the {}.",
]

CAR_TEMPLATES_EN = [
    "a clean origami {}.",
    "a photo of a {}.",
    "This is a photo of a {}",
    "There is a {} in the scene",
    "There is the {} in the scene",
    "a photo of a {} in the scene",
    "a photo of a small {}.",
    "a photo of a medium {}.",
    "a photo of a large {}.",
    "This is a photo of a small {}.",
    "This is a photo of a medium {}.",
    "This is a photo of a large {}.",
    "There is a small {} in the scene.",
    "There is a medium {} in the scene.",
    "There is a large {} in the scene.",
]

SUB_IMAGENET_TEMPLATES_EN = [
    "itap of a {}.",
    "a bad photo of a {}.",
    "a origami {}.",
    "a photo of the large {}.",
    "a {} in a video game.",
    "art of the {}.",
    "a photo of the small {}.",
]

MASKCLIP_TEMPLATES_EN = [
    "there is a {} in the scene.",
    "there is the {} in the scene.",
    "this is a {} in the scene.",
    "this is the {} in the scene.",
    "this is one {} in the scene.",  # maskclip
]

def build_prompts(class_name: str, dataset_type: str, templates: Optional[List[str]] = None) -> List[str]:
    """
    Builds a list of descriptive prompts for a given class name using predefined templates.

    This function takes a class name (e.g., "dog", "car") and formats it into multiple
    textual templates to create varied prompts. Averaging the text embeddings over several
    prompt variations gives a more stable class embedding for CLIP-based text encoders.

    The template set is chosen per evaluation protocol. This is the prompt configuration
    used for all numbers reported in the paper.

    Args:
        class_name (str): The name of the visual class (e.g., "dog", "car") for which to build prompts.
        dataset_type (str): Evaluation protocol name, one of {pascal_voc, voc20, pascal_context,
                            context59, coco_object, coco_stuff, cityscapes, ade20k}.
        templates (Optional[List[str]]): Explicit template list overriding the per-protocol
                                         selection. Defaults to None.

    Returns:
        List[str]: A list of formatted prompt strings ready for text encoding.
    """
    if templates is not None:
        template_list = templates
    elif dataset_type in ["pascal_context", "context59", "coco_object", "coco_stuff"]:
        template_list = OPENAI_TEMPLATES_EN
    elif dataset_type in ["pascal_voc", "voc20", "cityscapes", "ade20k"]:
        template_list = CAR_TEMPLATES_EN
    else:
        template_list = DEFAULT_TEMPLATES_EN

    # Format each template with the class name
    return [template.format(class_name) for template in template_list]


# -------------------------
# Text Encoder (CLIP via open_clip)
# -------------------------

class CLIPTextEncoder(nn.Module):
    """
    Frozen CLIP text encoder via open_clip for text feature extraction.

    This module loads a pre-trained CLIP text encoder using the `open_clip` library.
    The encoder is frozen (parameters not updated during training) and provides
    L2-normalized text embeddings. It's designed for encoding class names or
    descriptive prompts into a shared embedding space with visual features.

    The class handles model loading, tokenization, and provides a simple interface
    for encoding text prompts with optional aggregation of multiple prompts.
    """

    def __init__(self, model_name: str = "ViT-L-14", pretrained: str = "laion2b_s32b_b82k", device: Optional[str] = None):
        """
        Initializes the CLIPTextEncoder with a pre-trained CLIP model.

        Args:
            model_name (str): The name of the CLIP model architecture to load.
                             Common options include "ViT-L-14", "ViT-B-16", etc.
                             Defaults to "ViT-L-14".
            pretrained (str): The name of the pretrained dataset/checkpoint to use.
                             Examples: "laion2b_s32b_b82k", "openai", "datacomp1b".
                             Defaults to "laion2b_s32b_b82k".
            device (Optional[str]): The device (e.g., "cuda", "cpu") to load the model onto.
                                   If None, automatically selects CUDA if available, otherwise CPU.
                                   Defaults to None.
        """
        super().__init__()

        # Import open_clip library with error handling
        try:
            import open_clip
        except Exception as e:
            raise ImportError("open_clip is required: pip install open_clip_torch") from e

        self.oc = open_clip  # Store reference to open_clip module
        # print(self.oc.list_pretrained())

        # Load model and tokenizer
        self.model, _, _ = self.oc.create_model_and_transforms(model_name, pretrained=pretrained)
        self.tokenizer = self.oc.get_tokenizer(model_name)

        # Set device and move model to device
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()

        # Freeze all parameters to prevent updates during training
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def encode(self, prompts: List[str], aggregate: str = "mean") -> torch.Tensor:
        """
        Encodes a list of text prompts into L2-normalized embeddings.

        The prompts are tokenized using CLIP's tokenizer, passed through the frozen
        text encoder, and L2-normalized. Multiple prompts can be aggregated by averaging
        their embeddings, which is useful for creating robust text representations.

        Args:
            prompts (List[str]): A list of text strings to encode. Each string should be
                               a descriptive prompt (e.g., "a photo of a dog").
            aggregate (str): How to aggregate embeddings from multiple prompts:
                           - "mean": Average all prompt embeddings (default)
                           - Any other value: Return individual embeddings

        Returns:
            torch.Tensor: The L2-normalized text embeddings.
                         - If aggregated: Shape `[1, D]` where D is embedding dimension
                         - If not aggregated: Shape `[N, D]` where N is number of prompts
        """
        # Tokenize prompts and move to device
        tokens = self.tokenizer(prompts).to(self.device)

        # Encode text through CLIP model
        text_embeddings = self.model.encode_text(tokens)

        # Apply L2 normalization for cosine similarity compatibility
        text_embeddings = l2norm(text_embeddings)

        # Aggregate embeddings if requested (typically done for multiple prompt variations)
        if aggregate == "mean":
            # Average across all prompts to get a single embedding per text concept
            return text_embeddings.mean(0, keepdim=True)
        else:
            # Return individual embeddings for each prompt
            return text_embeddings


# -------------------------
# DINOv3 Vision Transformer Backbone (Hugging Face)
# -------------------------

class DinoV3HFBackbone(nn.Module):
    """
    Hugging Face DINOv3 wrapper for extracting dense patch-grid features.

    This module loads a pre-trained DINOv3 Vision Transformer model from Hugging Face
    and provides methods for image preprocessing and dense feature extraction. The
    model parameters are frozen for feature extraction only. It outputs a dense
    grid of patch features that can be used for downstream tasks like semantic segmentation.

    Key Features:
    - Automatic image preprocessing using Hugging Face transformers
    - Dense patch-grid feature extraction (removes CLS and register tokens)
    - Configurable input image size (automatically adjusted to patch size multiple)
    - Frozen parameters for efficient feature extraction
    """

    def __init__(self, model_id: str = "facebook/dinov3-vitl16-pretrain-lvd1689m", device: Optional[str] = None, image_size: int = 224):
        """
        Initializes the DinoV3HFBackbone with a pre-trained DINOv3 model.

        Args:
            model_id (str): The Hugging Face model ID for DINOv3.
                           Defaults to "facebook/dinov3-vitl16-pretrain-lvd1689m" (ViT-Large).
            device (Optional[str]): The device (e.g., "cuda", "cpu") to load the model onto.
                                   If None, automatically selects CUDA if available, otherwise CPU.
                                   Defaults to None.
            image_size (int): The target input image size for the DINOv3 model.
                             Must be divisible by the patch size (16 for most DINOv3 models).
                             Will be automatically adjusted if not divisible by 16.
                             Defaults to 224.
        """
        super().__init__()

        # Check for required dependencies
        if not _HAS_HF:
            raise ImportError("transformers not installed: pip install transformers")

        # Load image processor for preprocessing.
        # DINOv3 is a gated repository: authenticate once with `huggingface-cli login`
        # (or export HF_TOKEN) and the credential is picked up automatically here.
        self.processor = AutoImageProcessor.from_pretrained(model_id)

        # Adjust input image size to be divisible by patch size (16)
        if image_size % 16 != 0:
            original_size = image_size
            image_size = (image_size // 16) * 16
            logger.warning(f"Image size {original_size} is not divisible by 16. Adjusting to {image_size}")

        # Configure processor for the adjusted image size
        self.processor.size = {"height": image_size, "width": image_size}
        logger.info(f"DINOv3 input size set to: {image_size}x{image_size}")

        # Load the DINOv3 model in evaluation mode
        self.model = AutoModel.from_pretrained(model_id).eval()

        # Set device and move model to device
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    @torch.no_grad()
    def preprocess(self, img: Image.Image) -> Dict[str, torch.Tensor]:
        """
        Preprocesses a PIL Image into tensor format suitable for DINOv3.

        Uses the Hugging Face AutoImageProcessor to apply standard preprocessing:
        resizing, normalization, and tensor conversion. This ensures consistent
        input formatting for the DINOv3 model.

        Args:
            img (Image.Image): The input PIL Image in RGB format.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the preprocessed pixel values.
                                   Key: "pixel_values" with shape `[1, 3, H, W]` where
                                   H and W are the configured image dimensions.
        """
        # Process image using Hugging Face processor
        batch = self.processor(images=img, return_tensors="pt")
        return {"pixel_values": batch["pixel_values"]}  # Shape: [1, 3, H, W]

    @torch.no_grad()
    def forward_grid(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts dense patch-grid features and CLS token from the DINOv3 backbone.

        The input pixel values are passed through the DINOv3 Vision Transformer.
        Both patch tokens and CLS token are extracted separately. Register tokens
        are excluded.

        Args:
            pixel_values (torch.Tensor): Preprocessed image pixel values.
                                       Shape: `[B, 3, H_in, W_in]` where B=batch size.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - patch_features: Dense patch-grid features.
                                 Shape: `[B, C, H_p, W_p]` where:
                                 - C = embedding dimension (1024 for ViT-Large)
                                 - H_p, W_p = patch grid dimensions (H_in//16, W_in//16)
                - cls_token: Global CLS token features.
                            Shape: `[B, C]`
        """
        # Move input to device
        pixel_values = pixel_values.to(self.device)

        # Forward pass through DINOv3 model
        output = self.model(pixel_values=pixel_values)
        last_hidden_state = output.last_hidden_state  # Shape: [B, 1(+R)+HW, C]

        B, N, C = last_hidden_state.shape

        # Get model configuration parameters
        num_register_tokens = getattr(self.model.config, "num_register_tokens", 0)
        patch_size = getattr(self.model.config, "patch_size", 16)

        # Calculate patch grid dimensions
        H_in, W_in = pixel_values.shape[-2:]
        H_p, W_p = H_in // patch_size, W_in // patch_size

        # Extract CLS token (index 0)
        cls_token = last_hidden_state[:, 0, :]  # Shape: [B, C]
        
        # Extract patch tokens (skip CLS and register tokens)
        # Register tokens are at indices 1 to (1 + num_register_tokens)
        patch_tokens = last_hidden_state[:, 1 + num_register_tokens:, :]  # Shape: [B, HW, C]

        # Reshape patch tokens to grid format: [B, C, H_p, W_p]
        patch_features = patch_tokens.transpose(1, 2).reshape(B, C, H_p, W_p)

        return patch_features, cls_token


