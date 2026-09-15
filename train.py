# DINOv3 Open-Vocabulary Semantic Segmentation Training Pipeline (Flow Version)
#
# This script defines the training pipeline for DINOv3 OV-SS, incorporating:
# - DINOv3-L16 as the backbone.
# - Flow-based feature alignment for continuous text-image matching.
# - Optional Swin Transformer-style spatial aggregation.
# - Optimal Transport (OT) loss with boundary awareness (OT-PL).
# - NCE loss for global contrastive learning.
# - TensorBoard for logging and checkpoint management.
#
# Key Features:
# - Uses Hugging Face Transformers for DINOv3 backbone and image preprocessing.
# - CLIP text encoder for text feature extraction.
# - Supports PASCAL VOC and generic segmentation datasets.
# - Flexible configuration via JSON file.
# - Checkpointing and resume functionality.

import os
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime
import shutil

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import numpy as np
import random
import time
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend (no tkinter needed)
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from processing.coco_stuff.coco_stuff_dataset import COCOStuffDataset
from processing.coco_object.coco_object_dataset import COCOObjectDataset
from processing.pascal_voc.pascal_voc_dataset import PascalVOCDataset
from processing.pascal_context.pascal_context_dataset import PascalContextDataset
from processing.cityscapes.cityscapes_dataset import CityscapesDataset
from processing.ade20k.ade20k_dataset import ADE20KDataset
from model import (
    DinoV3HFBackbone, CLIPTextEncoder, TextCondHead, FlowTrainer
)
from model.components import build_prompts, l2norm
from utils.metrics import MeanIoU
from utils.model_utils import count_parameters

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DINOde-TRAIN")

# Model implementation file, snapshotted into <output_dir>/code_backup for reproducibility.
MODEL_FILE = "dinode.py"

class TrainingPipeline:
    """DINOv3 OV-SS Training Pipeline (Flow Version).
    
    This class orchestrates the training process for DINOv3-based open-vocabulary
    semantic segmentation models. It handles model initialization, training and
    validation loops, checkpointing, and configuration management.
    It supports training on PASCAL VOC and other custom segmentation datasets.
    """
    def __init__(self, config: Dict, output_dir: str = "outputs", num_classes: int = 182,
                 no_cache: bool = False):
        """
        Initializes the TrainingPipeline.

        Sets up training configurations, creates output directories, saves the config,
        determines the device (CUDA/CPU), and initializes the model components,
        optimizer, and learning rate scheduler.

        Args:
            config (Dict): A dictionary containing training and model configurations.
            output_dir (str): The base output directory for saving checkpoints and logs.
                              Defaults to "outputs".
            num_classes (int): Number of classes for mIoU calculation. Defaults to 182 for COCO-Stuff.
            no_cache (bool): If True, skip materializing the DINOv3 dense-feature caches and
                             compute backbone features on the fly instead. Trades speed for
                             disk space; see the caching section of README.md. Defaults to False.
        """
        self.config = config
        self.no_cache = no_cache
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config to output dir
        with open(self.output_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        # Save source code files for reproducibility
        self._backup_source_code()

        self.device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")
        self.writer = SummaryWriter(log_dir=self.output_dir / "logs")
        
        # Initialize single log file for all metrics
        self.log_file = self.output_dir / "training_log.txt"
        
        # Write header
        with open(self.log_file, "w") as f:
            f.write("="*80 + "\n")
            f.write("Training Log\n")
            f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
        
        logger.info(f"Training logs will be saved to: {self.log_file}")
        
        self._init_models()
        self.optimizer = self.trainer.opt
        self.scheduler = self.trainer.scheduler
        # Caching text features is a significant optimization for datasets with a
        # fixed vocabulary, as it avoids re-encoding the same class names on every step.
        
        param_counts = count_parameters(self.head)
        logger.info(f"Model Head Parameters: {param_counts}")
        self.writer.add_text("model", f"Total Parameters: {param_counts['total_params']}\nTrainable Parameters: {param_counts['trainable_params']}")
        
        logger.info(f"Pipeline initialized. Device: {self.device}")
    
    def _backup_source_code(self):
        """Snapshot the code used for this run, for experiment reproducibility."""
        code_backup_dir = self.output_dir / "code_backup"
        code_backup_dir.mkdir(parents=True, exist_ok=True)

        current_file = Path(__file__).resolve()
        project_root = current_file.parent

        files_to_backup = [
            # the entry-point script itself
            current_file,
            # core model files
            project_root / "model" / "__init__.py",
            project_root / "model" / MODEL_FILE,
            project_root / "model" / "components.py",
        ]

        backed_up_files = []
        for src_file in files_to_backup:
            if src_file.exists():
                # files from the model package keep their subdirectory
                if "model" in src_file.parts:
                    dest_dir = code_backup_dir / "model"
                    dest_dir.mkdir(exist_ok=True)
                    dest_file = dest_dir / src_file.name
                else:
                    dest_file = code_backup_dir / src_file.name

                shutil.copy2(src_file, dest_file)
                backed_up_files.append(src_file.name)
                logger.info(f"Backed up: {src_file.name}")
            else:
                logger.warning(f"File not found: {src_file}")

        backup_info = {
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "files_backed_up": backed_up_files,
            "model_file": MODEL_FILE,
            "entry_script": current_file.name,
        }

        with open(code_backup_dir / "backup_info.json", "w") as f:
            json.dump(backup_info, f, indent=2)

        logger.info(f"Source code backed up to: {code_backup_dir}")

    def _init_models(self):
        """Initializes the DINOv3 backbone, TextCondHead, and CLIPTextEncoder.
        
        Loads the DINOv3 backbone from Hugging Face, the `TextCondHead` with
        configured flow alignment, and the CLIP text encoder.
        It also sets up the `FlowTrainer` to manage the training steps and losses.
        """
        backbone_config = self.config["backbone"]
        head_config = self.config["head"]
        flow_config = self.config.get("flow", {})
        text_encoder_config = self.config["text_encoder"]
        
        self.backbone = DinoV3HFBackbone(
            model_id=backbone_config["model_id"],
            device=str(self.device),
            image_size=backbone_config["image_size"]
        )
        
        self.text_encoder = CLIPTextEncoder(
            model_name=text_encoder_config["model_name"],
            pretrained=text_encoder_config["pretrained"],
            device=str(self.device)
        )
        logger.info(f"Text encoder: CLIP ({text_encoder_config['model_name']} / {text_encoder_config['pretrained']})")
        
        self.head = TextCondHead(
            in_channels=head_config["in_channels"],
            tau=head_config["tau"],
            use_text_flow=head_config.get("use_text_flow", True),
            text_flow_steps=flow_config.get("steps", 10),
            text_flow_depth=flow_config.get("depth", 4),
            text_flow_dt=flow_config.get("dt"),  # Use config dt if provided
            use_cls_flow=head_config.get("use_cls_flow", True),
            use_cls_mlp=head_config.get("use_cls_mlp", False),
            topk=head_config.get("topk", 20),
            debug_interval=head_config.get("debug_interval", 100)
        ).to(self.device)
        
        # Log flow model status
        if self.head.use_text_flow:
            logger.info("✅ Text Flow Model: ENABLED")
            logger.info(f"   - Text flow steps (train): {self.head.text_flow_steps}")
            logger.info(f"   - Time step (dt): {self.head.text_flow_dt:.4f}")
            logger.info(f"   - Flow depth: {self.head.text_flow_depth}")
        else:
            logger.info("❌ Text Flow Model: DISABLED (using simple MLP)")
        
        if self.head.use_cls_flow:
            logger.info("✅ CLS Token Flow: ENABLED")
            logger.info(f"   - CLS flow steps: {self.head.cls_flow_steps}")
            logger.info(f"   - Time step (dt): {self.head.cls_flow_dt:.4f}")
        else:
            logger.info("❌ CLS Token Flow: DISABLED")

        self.trainer = FlowTrainer(
            backbone=self.backbone,
            head=self.head,
            txt=self.text_encoder,
            lr=self.config["training"]["learning_rate"],
            wd=self.config["training"]["weight_decay"],
            T_max=1, # Will be updated later
            w_nce=self.config["training"]["w_nce"],
            w_rf=self.config["training"]["w_rf"],
            grad_clip=self.config["training"]["grad_clip"],
        )
        
    def _init_optimizer_and_scheduler(self):
        """Initializes the optimizer and learning rate scheduler."""
        self.optimizer = self.trainer.optimizer
        self.scheduler = self.trainer.scheduler

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        self.trainer.head.train()
        metrics_sum = {}
        num_batches = len(dataloader)
        log_interval = self.config["logging"]["log_interval"]

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.config['training']['num_epochs']}")
        for batch_idx, batch in enumerate(pbar):
            class_names_list = batch['class_names']
            captions_list = batch.get('captions', [])
            dataset_indices = batch.get('dataset_idx', None)

            # Lookup cached caption CLIP embeddings by dataset index
            caption_clip_embeddings = None
            if dataset_indices is not None and self.caption_clip_cache_tensor is not None:
                idx_tensor = torch.tensor(dataset_indices, dtype=torch.long)
                caption_clip_embeddings = self.caption_clip_cache_tensor[idx_tensor]  # [B, 768]

            # Get cached DINOv3 features from dataloader (per-image cache)
            cached_backbone_features = None
            dino_A = batch.get('dino_A')
            dino_cls = batch.get('dino_cls')
            if dino_A is not None:
                cached_backbone_features = (dino_A.to(self.device), dino_cls.to(self.device))

            # Only transfer images to GPU if backbone cache is not available
            images = batch['image'].to(self.device) if cached_backbone_features is None else batch['image']

            metrics = self.trainer.train_step(
                images=images,
                name_lists=class_names_list,
                grid=self.config["training"]["grid_size"],
                iteration=epoch * num_batches + batch_idx,
                masks=None,
                captions=captions_list if any(captions_list) else None,
                caption_clip_embeddings=caption_clip_embeddings,
                cached_backbone_features=cached_backbone_features,
            )
            
            for key, value in metrics.items():
                if isinstance(value, torch.Tensor):
                    metrics_sum[key] = metrics_sum.get(key, 0.0) + value.item()
            
            if (batch_idx + 1) % log_interval == 0:
                avg_metrics = {k: v / log_interval for k, v in metrics_sum.items()}
                pbar.set_postfix(avg_metrics)
                
                # Log to TensorBoard
                for k, v in avg_metrics.items():
                    self.writer.add_scalar(f"train/{k}", v, epoch * num_batches + batch_idx)
                
                # Log to text file (fancy format)
                current_lr = self.optimizer.param_groups[0]['lr']
                with open(self.log_file, "a") as f:
                    f.write(f"Epoch [{epoch+1:3d}] Iter [{batch_idx+1:4d}], "
                           f"Loss: {avg_metrics.get('loss', 0.0):.6f}, "
                           f"L_nce: {avg_metrics.get('L_nce', 0.0):.6f}, "
                           f"L_rf: {avg_metrics.get('L_rf', 0.0):.6f}, "
                           f"L_vel: {avg_metrics.get('L_vel', 0.0):.6f}, "
                           f"L_vel_div: {avg_metrics.get('L_vel_div', 0.0):.6f}, "
                           f"L_vel_smooth: {avg_metrics.get('L_vel_smooth', 0.0):.6f}, "
                           f"LR: {current_lr:.8f}\n")
                
                metrics_sum = {k: 0.0 for k in metrics_sum}

        avg_epoch_metrics = {k: (v / num_batches) for k, v in metrics_sum.items() if k in metrics}
        return avg_epoch_metrics

    def save_head_to_ckpt(self, head: nn.Module, path: str):
        """Saves the state dictionary of the model's head to a specified path.
        
        This function is a utility to save only the `TextCondHead` component,
        which is useful for subsequent inference tasks without needing the entire
        training pipeline state.

        Args:
            head (nn.Module): The `TextCondHead` module to save.
            path (str): The file path where the head's state dictionary will be saved.
        """
        # This can be simplified to just save the state_dict
        torch.save(head.state_dict(), path)
        logger.info(f"Head checkpoint saved to {path}")

    def _sliding_window_inference(
        self,
        pil_image: Image.Image,
        head_fn,
        crop_size: int = 448,
        stride: int = 224,
        max_long_side: int = 2048,
    ) -> torch.Tensor:
        """Perform sliding window inference on a single PIL image.

        Resizes the image keeping aspect ratio: the shorter side becomes
        `crop_size` unless that would push the longer side beyond `max_long_side`,
        in which case the longer side is clamped to `max_long_side` instead.
        A `crop_size x crop_size` window is then slid with the given `stride`
        over the resized image, and the per-crop logits are accumulated and averaged.

        Args:
            pil_image: Input PIL image (RGB).
            head_fn: Callable that accepts backbone patch features A [1,C,h,w]
                     and returns logits [1,K,h,w].
            crop_size: Size of each sliding window crop (default 448).
            stride: Step size between consecutive crops (default 224).
            max_long_side: Maximum size allowed for the longer side after
                           resizing (default 2048).

        Returns:
            Averaged logit map, shape [K, H_resized, W_resized].
        """
        W_orig, H_orig = pil_image.size  # PIL gives (W, H)

        # Step 1: Resize — short side = crop_size, long side ≤ max_long_side
        scale = crop_size / min(H_orig, W_orig)
        new_H = int(round(H_orig * scale))
        new_W = int(round(W_orig * scale))
        if max(new_H, new_W) > max_long_side:
            scale = max_long_side / max(H_orig, W_orig)
            new_H = int(round(H_orig * scale))
            new_W = int(round(W_orig * scale))

        # Round to multiples of the ViT patch size (16) — required by DINOv3
        new_H = max(crop_size, (new_H // 16) * 16)
        new_W = max(crop_size, (new_W // 16) * 16)
        resized_img = pil_image.resize((new_W, new_H), Image.BILINEAR)
        
        # Step 2: Normalise using the same mean/std as the DINOv3 processor
        processor = self.backbone.processor
        mean = torch.tensor(processor.image_mean, dtype=torch.float32).view(3, 1, 1)
        std  = torch.tensor(processor.image_std,  dtype=torch.float32).view(3, 1, 1)
        img_np = np.array(resized_img, dtype=np.float32) / 255.0   # [H, W, 3]
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)     # [3, H, W]
        img_tensor = (img_tensor - mean) / std                      # normalise

        # Step 3: Compute crop start positions covering the full image
        def _positions(length):
            pos = list(range(0, length - crop_size + 1, stride))
            if not pos or pos[-1] + crop_size < length:
                pos.append(length - crop_size)
            return pos

        h_positions = _positions(new_H)
        w_positions = _positions(new_W)

        # Step 4: Slide and accumulate logits
        logits_sum = None
        count_map  = None

        for h_start in h_positions:
            for w_start in w_positions:
                crop = img_tensor[
                    :,
                    h_start : h_start + crop_size,
                    w_start : w_start + crop_size,
                ].unsqueeze(0).to(self.device)                # [1, 3, crop_size, crop_size]

                with torch.no_grad():
                    A, _ = self.backbone.forward_grid(crop)   # [1, C, h_p, w_p]
                    logits_crop = head_fn(A)                  # [1, K, h_p, w_p]

                logits_up = F.interpolate(
                    logits_crop,
                    size=(crop_size, crop_size),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)                                   # [K, crop_size, crop_size]

                if logits_sum is None:
                    K = logits_up.shape[0]
                    logits_sum = torch.zeros((K, new_H, new_W), dtype=torch.float32, device=self.device)
                    count_map  = torch.zeros((1, new_H, new_W), dtype=torch.float32, device=self.device)

                logits_sum[:, h_start : h_start + crop_size, w_start : w_start + crop_size] += logits_up
                count_map[:,  h_start : h_start + crop_size, w_start : w_start + crop_size] += 1

        logits_avg = logits_sum / (count_map + 1e-6)           # [K, new_H, new_W]
        return logits_avg

    def _sliding_window_inference_from_cache(
        self,
        cached: dict,
        head_fn,
    ) -> torch.Tensor:
        """Sliding window inference using pre-computed DINOv3 crop features.

        Each entry in the cache corresponds to a 448×448 crop of the resized
        image that was already passed through the DINOv3 backbone.  Only the
        head (which changes every epoch) is run here.

        Args:
            cached: Dict loaded from a .npy slide-window cache file:
                    'features'    float16 [N_crops, C, h_p, w_p]
                    'h_positions' int32   [N_h]   — top-left row of each crop
                    'w_positions' int32   [N_w]   — top-left col of each crop
                    'new_H', 'new_W'              — resized image dimensions
                    'crop_size'                   — size of each crop (e.g. 448)
            head_fn: Callable A [1,C,h_p,w_p] -> logits [1,K,h_p,w_p].

        Returns:
            Averaged logit map, shape [K, new_H, new_W].
        """
        features    = torch.from_numpy(cached['features'].astype(np.float32)).to(self.device)
        h_positions = cached['h_positions'].tolist()
        w_positions = cached['w_positions'].tolist()
        new_H       = int(cached['new_H'])
        new_W       = int(cached['new_W'])
        crop_size   = int(cached['crop_size'])

        logits_sum = None
        count_map  = None

        with torch.no_grad():
            crop_idx = 0
            for h_start in h_positions:
                for w_start in w_positions:
                    A           = features[crop_idx].unsqueeze(0)  # [1, C, h_p, w_p]
                    logits_crop = head_fn(A)                        # [1, K, h_p, w_p]

                    logits_up = F.interpolate(
                        logits_crop,
                        size=(crop_size, crop_size),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)                                    # [K, crop_size, crop_size]

                    if logits_sum is None:
                        K = logits_up.shape[0]
                        logits_sum = torch.zeros((K, new_H, new_W), dtype=torch.float32, device=self.device)
                        count_map  = torch.zeros((1, new_H, new_W), dtype=torch.float32, device=self.device)

                    logits_sum[:, h_start:h_start+crop_size, w_start:w_start+crop_size] += logits_up
                    count_map[:,  h_start:h_start+crop_size, w_start:w_start+crop_size] += 1
                    crop_idx += 1

        return logits_sum / (count_map + 1e-6)                     # [K, new_H, new_W]

    def validate(self, dataloader: DataLoader, epoch: int, dataset_type: str = "coco_stuff", save_visualizations: bool = False, infer_steps_list: Optional[List[int]] = None) -> Dict[str, float]:
        self.head.eval()
        self.backbone.eval()

        # Sliding window inference configuration
        slide_cfg = self.config["validation"].get("slide_window", {})
        use_slide_window = slide_cfg.get("enabled", False)
        slide_crop_size = slide_cfg.get("crop_size", 448)
        slide_stride = slide_cfg.get("stride", 224)
        slide_max_long_side = slide_cfg.get("max_long_side", 2048)
        if use_slide_window:
            logger.info(
                f"[Slide Window] crop={slide_crop_size}, stride={slide_stride}, "
                f"max_long={slide_max_long_side}"
            )

        use_pamr = self.config.get("validation", {}).get("use_pamr", False)
        if use_pamr:
            logger.info("[PAMR] Post-processing enabled (iter=10, dilations=[1,2,4,8,12,24])")

        # Define dataset configurations for extensibility
        # 8 Evaluation Protocols:
        # BG Include: VOC21 (pascal_voc), Context60 (pascal_context), COCO Object (coco_object)
        # BG Exclude: VOC20 (voc20), Context59 (context59), COCO Stuff (coco_stuff), Cityscapes (cityscapes), ADE20K (ade20k)
        DATASET_INFO = {
            # === BG INCLUDE PROTOCOLS ===
            "pascal_voc": {  # VOC21 - 21 classes with background
                "num_classes": 21,
                "use_threshold": True,
                "thresholds": [0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29],
                "has_background": True,
                "include_background_in_miou": True,
                "reduce_zero_label": False,
                "requires_label_mapping": False,
            },
            "pascal_context": {  # Context60 - 60 classes with background
                "num_classes": 60,
                "use_threshold": True,
                "thresholds": [0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14],
                "has_background": True,
                "include_background_in_miou": True,
                "reduce_zero_label": False,
                "requires_label_mapping": False,
            },
            "coco_object": {  # COCO Object - 81 classes with background
                "num_classes": 81,
                "use_threshold": True,
                "thresholds": [0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14],
                "has_background": True,
                "include_background_in_miou": True,
                "reduce_zero_label": False,
                "requires_label_mapping": False,
            },
            # === BG EXCLUDE PROTOCOLS ===
            "voc20": {  # VOC20 - 20 FG classes only (BG excluded via reduce_zero_label)
                "num_classes": 20,  # 20 FG classes only; GT 0→255 by reduce_zero_label
                "use_threshold": False,
                "thresholds": [0.0001],
                "has_background": False,
                "include_background_in_miou": True,  # All 20 classes are FG
                "reduce_zero_label": True,            # GT 0→255(ignore), FG labels shift -1
                "requires_label_mapping": False,
            },
            "context59": {  # Context59 - 59 FG classes only (BG excluded via reduce_zero_label)
                "num_classes": 59,  # 59 FG classes only; GT 0→255 by reduce_zero_label
                "use_threshold": False,
                "thresholds": [0.0001],
                "has_background": False,
                "include_background_in_miou": True,  # All 59 classes are FG
                "reduce_zero_label": True,            # GT 0→255(ignore), FG labels shift -1
                "requires_label_mapping": False,
            },
            "coco_stuff": {  # COCO-Stuff 171 - 171 classes, NO background
                "num_classes": 171,  # 171 stuff classes (BG excluded)
                "use_threshold": False,
                "thresholds": [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09],
                "has_background": False,
                "include_background_in_miou": True,  # All classes included (no BG exists)
                "reduce_zero_label": False,
                "requires_label_mapping": False,
            },
            "cityscapes": {  # Cityscapes - 19 classes, NO background
                "num_classes": 19,
                "use_threshold": False,
                "thresholds": [0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14],
                "has_background": False,
                "include_background_in_miou": True,  # All classes included (no BG exists)
                "reduce_zero_label": False,
                "requires_label_mapping": False,
            },
            "ade20k": {  # ADE20K - 150 FG classes only (BG excluded via reduce_zero_label)
                "num_classes": 150,  # 150 FG classes only; GT 0→255 by reduce_zero_label
                "use_threshold": False,
                "thresholds": [0.0001],
                "has_background": False,
                "include_background_in_miou": True,  # All 150 classes are FG
                "reduce_zero_label": True,            # GT 0→255(ignore), FG labels shift -1
                "requires_label_mapping": False,
            },
        }
        
        # Get dataset configuration
        dataset_config = DATASET_INFO.get(dataset_type, {
            "num_classes": 151,  # Default to ADE20K (151 classes)
            "use_threshold": True,
            "thresholds": [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09],
            "has_background": True,
            "include_background_in_miou": False,
            "requires_label_mapping": False,
        })

        num_classes = dataset_config["num_classes"]
        include_bg_in_miou = dataset_config.get("include_background_in_miou", True)

        # reduce_zero_label: for BG-Exclude protocols, GT 0→255 before metric update.
        # After that transform, only 255 needs to be ignored (BG is already 255).
        reduce_zero_label = dataset_config.get("reduce_zero_label", False)

        # Helper functions applied to GT masks when reduce_zero_label is True.
        def _rzl_np(m: np.ndarray) -> np.ndarray:
            """GT 0(BG)→255(ignore), valid FG labels -= 1."""
            out = m.copy()
            valid = (m != 0) & (m != 255)
            out[m == 0] = 255
            out[valid] -= 1
            return out

        def _rzl_tensor(m: torch.Tensor) -> torch.Tensor:
            """GT 0(BG)→255(ignore), valid FG labels -= 1."""
            valid = (m != 0) & (m != 255)
            out = m.clone()
            out[m == 0] = 255
            out[valid] -= 1
            return out

        # Create a new MeanIoU metric with the correct num_classes and BG handling
        # For all protocols ignore_indices=[255]: RZL converts BG(0)→255 for BG-Exclude,
        # so we don't need to ignore 0 explicitly after the transform.
        ignore_indices = [255]
        miou_metric = MeanIoU(
            num_classes=num_classes,
            device=self.device,
            ignore_indices=ignore_indices,
            include_background=include_bg_in_miou
        )
        
        # Get threshold configuration
        use_threshold = dataset_config["use_threshold"]
        thresholds = dataset_config.get("thresholds", [0.5])
        threshold_results = {}
        all_predictions = []  # Store predictions for all thresholds
        all_masks = []  # Store ground truth masks
        
        with torch.no_grad():
            # IMPORTANT: Re-encode text features with the updated text_proj weights!
            # The text_proj is trained during training, so we need to re-encode the class names
            # with the current text_proj weights for accurate validation.
            
            # Get class names for text encoding based on dataset type and BG mode
            # For BG include datasets: include background in class names
            # For BG exclude datasets: exclude background from class names

            if dataset_type == "pascal_voc":  # VOC21 - BG include
                class_names = [
                    'ground, land, grass, tree, building, wall, sky, lake, water, river, sea, railway, railroad, keyboard, helmet, cloud, house, mountain, ocean, road, rock, street, valley, bridge, sign', 
                    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                    'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
                    'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
                ]
            elif dataset_type == "voc20":  # VOC20 - BG exclude (reduce_zero_label: 20 FG classes)
                class_names = [
                    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                    'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
                    'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
                ]
            elif dataset_type == "pascal_context":  # Context60 - BG include
                class_names = [
                    'background', 'aeroplane', 'bag', 'bed', 'bedclothes', 'bench', 'bicycle', 'bird',
                    'boat', 'book', 'bottle', 'building', 'bus', 'cabinet', 'car', 'cat', 'ceiling',
                    'chair', 'cloth', 'computer', 'cow', 'cup', 'curtain', 'dog', 'door', 'fence',
                    'floor', 'flower', 'food', 'grass', 'ground', 'horse', 'keyboard', 'light',
                    'motorbike', 'mountain', 'mouse', 'person', 'plate', 'platform', 'pottedplant',
                    'road', 'rock', 'sheep', 'shelves', 'sidewalk', 'sign', 'sky', 'snow', 'sofa',
                    'table', 'track', 'train', 'tree', 'truck', 'tvmonitor', 'wall', 'water', 'window', 'wood'
                ]
            elif dataset_type == "context59":  # Context59 - BG exclude (reduce_zero_label: 59 FG classes)
                class_names = [
                    'aeroplane', 'bag', 'bed', 'bedclothes', 'bench', 'bicycle', 'bird',
                    'boat', 'book', 'bottle', 'building', 'bus', 'cabinet', 'car', 'cat', 'ceiling',
                    'chair', 'cloth', 'computer', 'cow', 'cup', 'curtain', 'dog', 'door', 'fence',
                    'floor', 'flower', 'food', 'grass', 'ground', 'horse', 'keyboard', 'light',
                    'motorbike', 'mountain', 'mouse', 'person', 'plate', 'platform', 'pottedplant',
                    'road', 'rock', 'sheep', 'shelves', 'sidewalk', 'sign', 'sky', 'snow', 'sofa',
                    'table', 'track', 'train', 'tree', 'truck', 'tvmonitor', 'wall', 'water', 'window', 'wood'
                ]
            elif dataset_type == "coco_object":  # COCO Object 81 - BG include
                class_names = [
                    'ground, land, grass, tree, building, wall, sky, lake, water, river, sea, railway, railroad, helmet, cloud, house, mountain, ocean, road, rock, street, valley, bridge', 
                    'person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train',
                    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter',
                    'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
                    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
                    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
                    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
                    'pizza', 'donut', 'cake', 'chair', 'couch', 'pottedplant', 'bed', 'diningtable',
                    'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
                ]
            elif dataset_type == "coco_stuff":  # COCO Stuff 171 - NO BG (BG excluded from dataset)
                # 171 classes - background is NOT included
                class_names = [
                    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                    'trafficlight', 'firehydrant', 'stopsign', 'parkingmeter', 'bench', 'bird', 'cat', 'dog',
                    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
                    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sportsball', 'kite',
                    'baseballbat', 'baseballglove', 'skateboard', 'surfboard', 'tennisracket', 'bottle',
                    'wineglass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
                    'orange', 'broccoli', 'carrot', 'hotdog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                    'pottedplant', 'bed', 'diningtable', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                    'keyboard', 'cellphone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
                    'clock', 'vase', 'scissors', 'teddybear', 'hairdrier', 'toothbrush', 'banner', 'blanket',
                    'branch', 'bridge', 'building-other', 'bush', 'cabinet', 'cage', 'cardboard', 'carpet',
                    'ceiling-other', 'ceiling-tile', 'cloth', 'clothes', 'clouds', 'counter', 'cupboard',
                    'curtain', 'desk-stuff', 'dirt', 'door-stuff', 'fence', 'floor-marble', 'floor-other',
                    'floor-stone', 'floor-tile', 'floor-wood', 'flower', 'fog', 'food-other', 'fruit',
                    'furniture-other', 'grass', 'gravel', 'ground-other', 'hill', 'house', 'leaves', 'light',
                    'mat', 'metal', 'mirror-stuff', 'moss', 'mountain', 'mud', 'napkin', 'net', 'paper',
                    'pavement', 'pillow', 'plant-other', 'plastic', 'platform', 'playingfield', 'railing',
                    'railroad', 'river', 'road', 'rock', 'roof', 'rug', 'salad', 'sand', 'sea', 'shelf',
                    'sky-other', 'skyscraper', 'snow', 'solid-other', 'stairs', 'stone', 'straw',
                    'structural-other', 'table', 'tent', 'textile-other', 'towel', 'tree', 'vegetable',
                    'wall-brick', 'wall-concrete', 'wall-other', 'wall-panel', 'wall-stone', 'wall-tile',
                    'wall-wood', 'water-other', 'waterdrops', 'window-blind', 'window-other', 'wood'
                ]
            elif dataset_type == "cityscapes":  # Cityscapes 19 - NO BG
                class_names = [
                    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
                    'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
                    'truck', 'bus', 'train', 'motorcycle', 'bicycle'
                ]
            elif dataset_type == "ade20k":  # ADE20K 150 - BG exclude (reduce_zero_label: 150 FG classes)
                # 150 FG classes - background excluded
                class_names = [
                    'wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed', 'windowpane',
                    'grass', 'cabinet', 'sidewalk', 'person', 'earth', 'door', 'table', 'mountain', 'plant',
                    'curtain', 'chair', 'car', 'water', 'painting', 'sofa', 'shelf', 'house', 'sea', 'mirror',
                    'rug', 'field', 'armchair', 'seat', 'fence', 'desk', 'rock', 'wardrobe', 'lamp', 'bathtub',
                    'railing', 'cushion', 'base', 'box', 'column', 'signboard', 'chestofdrawers', 'counter',
                    'sand', 'sink', 'skyscraper', 'fireplace', 'refrigerator', 'grandstand', 'path', 'stairs',
                    'runway', 'case', 'pooltable', 'pillow', 'screendoor', 'stairway', 'river', 'bridge',
                    'bookcase', 'blind', 'coffeetable', 'toilet', 'flower', 'book', 'hill', 'bench',
                    'countertop', 'stove', 'palm', 'kitchenisland', 'computer', 'swivelchair', 'boat', 'bar',
                    'arcademachine', 'hovel', 'bus', 'towel', 'light', 'truck', 'tower', 'chandelier', 'awning',
                    'streetlight', 'booth', 'televisionreceiver', 'airplane', 'dirttrack', 'apparel', 'pole',
                    'land', 'bannister', 'escalator', 'ottoman', 'bottle', 'buffet', 'poster', 'stage', 'van',
                    'ship', 'fountain', 'conveyerbelt', 'canopy', 'washer', 'plaything', 'swimmingpool', 'stool',
                    'barrel', 'basket', 'waterfall', 'tent', 'bag', 'minibike', 'cradle', 'oven', 'ball', 'food',
                    'step', 'tank', 'tradename', 'microwave', 'pot', 'animal', 'bicycle', 'lake', 'dishwasher',
                    'screen', 'blanket', 'sculpture', 'hood', 'sconce', 'vase', 'trafficlight', 'tray', 'ashcan',
                    'fan', 'pier', 'crtscreen', 'plate', 'monitor', 'bulletinboard', 'shower', 'radiator',
                    'glass', 'clock', 'flag'
                ]
            else:
                # Fallback: Get class names from dataset
                class_names = dataloader.dataset.get_class_names()

            # Multi-word BG class: if class_names[0] contains commas, split into individual
            # sub-class names. All BG logit channels are max-pooled back to 1 before softmax.
            if dataset_config.get("has_background", False) and class_names and ',' in class_names[0]:
                _bg_words = [w.strip() for w in class_names[0].split(',')]
                _n_bg_channels = len(_bg_words)
                _flat_class_names = _bg_words + class_names[1:]
                logger.info(
                    f"[{dataset_type.upper()}] Multi-word BG: {_n_bg_channels} BG sub-classes "
                    f"→ {len(_flat_class_names)} total logits, max-pooled to {len(class_names)} channels"
                )
            else:
                _n_bg_channels = 1
                _flat_class_names = class_names

            # Ensure per-image DINOv3 cache exists for this validation dataset
            data_dir = self.val_data_dirs.get(dataset_type)
            if data_dir is not None and not self.no_cache:
                backbone_cfg = self.config["backbone"]
                backbone_id = backbone_cfg["model_id"].replace("/", "-")
                val_image_size = self.config.get("data", {}).get("validation_image_size", backbone_cfg["image_size"])
                val_dino_cache_dir = Path(data_dir) / f"dino_feature_{backbone_id}_{val_image_size}" / "val"

                N = len(dataloader.dataset)
                existing = len(list(val_dino_cache_dir.glob("*.npy"))) if val_dino_cache_dir.exists() else 0

                if existing >= N:
                    logger.info(f"Val DINOv3 per-image cache complete for {dataset_type} ({existing} files)")
                else:
                    val_dino_cache_dir.mkdir(parents=True, exist_ok=True)
                    logger.info(f"Creating val DINOv3 per-image cache for {dataset_type} ({existing}/{N} exist)...")

                    # Get raw data array from dataset
                    raw_data = getattr(dataloader.dataset, 'processed_data', None)
                    if raw_data is None:
                        raw_data = getattr(dataloader.dataset, 'data', None)

                    processor = self.backbone.processor
                    original_size = processor.size
                    processor.size = {"height": val_image_size, "width": val_image_size}

                    cache_batch_size = 256
                    for start_idx in tqdm(range(0, N, cache_batch_size), desc=f"Caching val DINOv3 features ({dataset_type})"):
                        end_idx = min(start_idx + cache_batch_size, N)

                        items_to_cache = []
                        for i in range(start_idx, end_idx):
                            img_name = os.path.splitext(os.path.basename(str(raw_data[i]['image_path'])))[0]
                            cache_file = val_dino_cache_dir / f"{img_name}.npy"
                            if not cache_file.exists():
                                items_to_cache.append((i, raw_data[i], img_name))

                        if not items_to_cache:
                            continue

                        pil_images = [Image.open(str(item[1]['image_path'])).convert('RGB') for item in items_to_cache]
                        processed = processor(images=pil_images, return_tensors="pt")
                        images_tensor = processed["pixel_values"].to(self.device)
                        with torch.no_grad():
                            A, cls_token = self.backbone.forward_grid(images_tensor)

                        A_np = A.cpu().numpy().astype(np.float16)
                        cls_np = cls_token.cpu().numpy().astype(np.float16)
                        for j, (_, _, img_name) in enumerate(items_to_cache):
                            np.save(str(val_dino_cache_dir / f"{img_name}.npy"), {'A': A_np[j], 'cls': cls_np[j]})

                    processor.size = original_size
                    logger.info(f"Saved val DINOv3 per-image cache for {dataset_type} to {val_dino_cache_dir}")

                # Ensure dataset knows cache dir for dataloader workers
                dataloader.dataset.dino_cache_dir = str(val_dino_cache_dir)

            # Ensure slide-window DINOv3 crop cache exists
            slide_cache_dir = None
            if use_slide_window and data_dir is not None and not self.no_cache:
                _bb_id = self.config["backbone"]["model_id"].replace("/", "-")
                _sw_dir_name = f"dino_slide_{_bb_id}_{slide_crop_size}x{slide_stride}_l{slide_max_long_side}"
                slide_cache_dir = Path(data_dir) / _sw_dir_name / "val"
                N_sw = len(dataloader.dataset)
                existing_sw = len(list(slide_cache_dir.glob("*.npy"))) if slide_cache_dir.exists() else 0

                if existing_sw >= N_sw:
                    logger.info(f"Slide-window DINOv3 cache complete for {dataset_type} ({existing_sw} files)")
                else:
                    slide_cache_dir.mkdir(parents=True, exist_ok=True)
                    logger.info(f"Creating slide-window DINOv3 cache for {dataset_type} ({existing_sw}/{N_sw} exist)...")
                    raw_data_sw = getattr(dataloader.dataset, 'processed_data', None)
                    if raw_data_sw is None:
                        raw_data_sw = getattr(dataloader.dataset, 'data', None)

                    _proc_sw = self.backbone.processor
                    _mean_sw = torch.tensor(_proc_sw.image_mean, dtype=torch.float32).view(3, 1, 1)
                    _std_sw  = torch.tensor(_proc_sw.image_std,  dtype=torch.float32).view(3, 1, 1)

                    def _sw_pos(length, csize, cstride):
                        pos = list(range(0, length - csize + 1, cstride))
                        if not pos or pos[-1] + csize < length:
                            pos.append(length - csize)
                        return pos

                    _sw_crop_batch = 32  # max crops per forward pass to avoid OOM
                    for idx_sw in tqdm(range(N_sw), desc=f"Caching slide-window features ({dataset_type})"):
                        cache_file_sw = slide_cache_dir / f"{idx_sw:06d}.npy"
                        if cache_file_sw.exists():
                            continue

                        pil_c = Image.open(str(raw_data_sw[idx_sw]['image_path'])).convert('RGB')
                        W_c, H_c = pil_c.size
                        sc = slide_crop_size / min(H_c, W_c)
                        nH = int(round(H_c * sc)); nW = int(round(W_c * sc))
                        if max(nH, nW) > slide_max_long_side:
                            sc = slide_max_long_side / max(H_c, W_c)
                            nH = int(round(H_c * sc)); nW = int(round(W_c * sc))
                        nH = max(slide_crop_size, (nH // 16) * 16)
                        nW = max(slide_crop_size, (nW // 16) * 16)
                        resized_c = pil_c.resize((nW, nH), Image.BILINEAR)
                        img_np_c = np.array(resized_c, dtype=np.float32) / 255.0
                        img_t_c = (torch.from_numpy(img_np_c).permute(2, 0, 1) - _mean_sw) / _std_sw

                        h_pos = _sw_pos(nH, slide_crop_size, slide_stride)
                        w_pos = _sw_pos(nW, slide_crop_size, slide_stride)
                        crops_list = [
                            img_t_c[:, h:h + slide_crop_size, w:w + slide_crop_size]
                            for h in h_pos for w in w_pos
                        ]

                        # Batch crops to avoid OOM
                        A_parts = []
                        for b_start in range(0, len(crops_list), _sw_crop_batch):
                            crops_t = torch.stack(crops_list[b_start:b_start + _sw_crop_batch]).to(self.device)
                            with torch.no_grad():
                                A_part, _ = self.backbone.forward_grid(crops_t)
                            A_parts.append(A_part.cpu())
                        A_c = torch.cat(A_parts, dim=0)

                        np.save(str(cache_file_sw), {
                            'features':    A_c.numpy().astype(np.float16),
                            'h_positions': np.array(h_pos, dtype=np.int32),
                            'w_positions': np.array(w_pos, dtype=np.int32),
                            'new_H':       np.int32(nH),
                            'new_W':       np.int32(nW),
                            'crop_size':   np.int32(slide_crop_size),
                        })

                    logger.info(f"Saved slide-window DINOv3 cache for {dataset_type} to {slide_cache_dir}")

            # Get or compute cached CLIP embeddings for this dataset (single-dataset cache)
            if not hasattr(self, "_val_clip_cache_dataset"):
                self._val_clip_cache_dataset = None
                self._val_clip_cache_tensor = None

            if self._val_clip_cache_dataset != dataset_type:
                self._val_clip_cache_dataset = dataset_type
                self._val_clip_cache_tensor = None

                data_dir = self.val_data_dirs.get(dataset_type)
                text_enc_cfg = self.config["text_encoder"]
                model_name = text_enc_cfg["model_name"].replace("/", "-")
                pretrained = text_enc_cfg["pretrained"]
                rzl_suffix = "_rzl" if reduce_zero_label else ""
                cache_filename = f"val_clip_cache_{dataset_type}_{model_name}_{pretrained}{rzl_suffix}.npy"

                loaded_from_file = False
                if data_dir is not None:
                    cache_path = Path(data_dir) / cache_filename
                    if cache_path.exists():
                        self._val_clip_cache_tensor = torch.from_numpy(np.load(str(cache_path))).to(self.device)
                        loaded_from_file = True
                        logger.info(f"Loaded val CLIP cache for {dataset_type} from {cache_path}")

                if not loaded_from_file:
                    logger.info(f"No val CLIP cache for {dataset_type}")
                    with torch.no_grad():
                        T_clip = torch.stack([
                            self.text_encoder.encode(build_prompts(n, dataset_type), aggregate="mean").squeeze(0)
                            for n in _flat_class_names
                        ], dim=0)  # [M+N, 768] where M=_n_bg_channels, N=n_fg_classes
                    self._val_clip_cache_tensor = T_clip
                    if data_dir is not None:
                        cache_path = Path(data_dir) / cache_filename
                        np.save(str(cache_path), T_clip.cpu().numpy())
                        logger.info(f"Saved val CLIP cache for {dataset_type} to {cache_path}")

            text_features = self.trainer._encode_texts(
                _flat_class_names, cached_clip_embeddings=self._val_clip_cache_tensor
            )
            logger.info(f"Re-encoded text features for {dataset_type} validation with updated text_proj (CLIP cached)")
            
            sample_count = 0
            for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Validation Epoch {epoch+1}")):

                if use_slide_window:
                    # ── Sliding-window path ─────────────────────────────────
                    pil_images_batch = batch.get('pil_images', [])
                    pil_masks_batch  = batch.get('pil_masks',  [])

                    head_fn = lambda A: self.head(A, text_features)[0]

                    for _sw_i, (pil_img, pil_mask) in enumerate(zip(pil_images_batch, pil_masks_batch)):
                        _sw_ds_idx = sample_count + _sw_i
                        _sw_cf = slide_cache_dir / f"{_sw_ds_idx:06d}.npy" if slide_cache_dir is not None else None
                        if _sw_cf is not None and _sw_cf.exists():
                            _sw_cached = np.load(str(_sw_cf), allow_pickle=True).item()
                            logits_map = self._sliding_window_inference_from_cache(_sw_cached, head_fn)
                        else:
                            logits_map = self._sliding_window_inference(
                                pil_img, head_fn,
                                crop_size=slide_crop_size,
                                stride=slide_stride,
                                max_long_side=slide_max_long_side,
                            )  # [K, H_resized, W_resized]

                        mask_np = np.array(pil_mask, dtype=np.int64)
                        if reduce_zero_label:
                            mask_np = _rzl_np(mask_np)
                        mask_single = torch.from_numpy(mask_np).unsqueeze(0)  # [1, H_orig, W_orig]

                        logits_up = F.interpolate(
                            logits_map.unsqueeze(0),
                            size=mask_single.shape[-2:],
                            mode='bilinear', align_corners=False,
                        )  # [1, K, H_orig, W_orig]

                        if _n_bg_channels > 1:
                            _bg_max = logits_up[:, :_n_bg_channels].max(dim=1, keepdim=True)[0]
                            logits_up = torch.cat([_bg_max, logits_up[:, _n_bg_channels:]], dim=1)
                        pred_probs = F.softmax(logits_up, dim=1)  # [1, K, H, W]

                        if use_pamr:
                            img_t = (
                                torch.from_numpy(np.array(pil_img, dtype=np.float32) / 255.0)
                                .permute(2, 0, 1).unsqueeze(0).to(self.device)
                            )  # [1, 3, H_orig, W_orig]
                            # Adaptive chunk size: keep [1, chunk, 48, H, W] under ~2 GB.
                            # PAMR is class-independent so any chunk size is mathematically equivalent.
                            _pH, _pW = pred_probs.shape[-2:]
                            _pamr_chunk = max(1, min(30, int(2e9 / (48 * _pH * _pW * 4))))
                            with torch.no_grad():
                                for _pc in range(0, pred_probs.shape[1], _pamr_chunk):
                                    pred_probs[:, _pc:_pc + _pamr_chunk] = self.trainer.apply_pamr(
                                        img_t, pred_probs[:, _pc:_pc + _pamr_chunk]
                                    )

                        pred_label = torch.argmax(pred_probs, dim=1)    # [1, H, W]
                        max_prob   = torch.max(pred_probs, dim=1)[0]    # [1, H, W]

                        if use_threshold:
                            all_predictions.append({
                                'pred_labels': pred_label.cpu(),       
                                'max_probs':   max_prob.cpu(),          
                                'images':      None,   # skip per-image visualisation in slide mode
                            })
                            all_masks.append(mask_single.cpu())        
                        else:
                            miou_metric.update(pred_label.cpu(), mask_single.cpu())

                    sample_count += len(pil_images_batch)

                else:
                    # ── Standard (whole-image) path ─────────────────────────
                    cached_backbone_features = None
                    dino_A  = batch.get('dino_A')
                    dino_cls = batch.get('dino_cls')
                    if dino_A is not None:
                        cached_backbone_features = (dino_A.to(self.device), dino_cls.to(self.device))

                    images = batch['image'].to(self.device) if cached_backbone_features is None else batch['image']
                    masks  = batch['mask']
                    if reduce_zero_label:
                        masks = _rzl_tensor(masks)

                    if cached_backbone_features is not None:
                        A, _ = cached_backbone_features
                    else:
                        A, _ = self.backbone.forward_grid(images)
                    logits, _, _ = self.head(A, text_features)

                    pred_masks      = F.interpolate(logits, size=masks.shape[-2:], mode='bilinear', align_corners=False)
                    if _n_bg_channels > 1:
                        _bg_max = pred_masks[:, :_n_bg_channels].max(dim=1, keepdim=True)[0]
                        pred_masks = torch.cat([_bg_max, pred_masks[:, _n_bg_channels:]], dim=1)
                    pred_probabilities = F.softmax(pred_masks, dim=1)
                    pred_labels_raw = torch.argmax(pred_masks, dim=1)
                    max_probabilities = torch.max(pred_probabilities, dim=1)[0]

                    if use_threshold:
                        all_predictions.append({
                            'pred_labels': pred_labels_raw.cpu(),
                            'max_probs':   max_probabilities.cpu(),
                            'images':      images if (save_visualizations and sample_count < 10) else None,
                        })
                        all_masks.append(masks.cpu())
                    else:
                        miou_metric.update(pred_labels_raw.cpu(), masks.cpu())

                    if save_visualizations and sample_count < 10:
                        if not hasattr(miou_metric, 'stored_images'):
                            miou_metric.stored_images     = []
                            miou_metric.stored_masks      = []
                            miou_metric.stored_predictions = []
                        if len(miou_metric.stored_images) < 10:
                            miou_metric.stored_images.append(images)
                            miou_metric.stored_masks.append(masks.cpu())
                            miou_metric.stored_predictions.append(pred_labels_raw.cpu())

                    sample_count += len(images)

        if use_threshold:
            # Evaluate all thresholds (only for Pascal VOC)
            logger.info(f"\n{'='*60}")
            logger.info(f"Background Threshold Comparison - {dataset_type.upper()} (Epoch {epoch+1})")
            logger.info(f"{'='*60}")
            
            best_threshold = None
            best_miou = 0.0
            previous_miou = None  # Track previous mIoU to detect decrease
            
            for threshold in thresholds:
                miou_metric.reset()
                
                # Apply threshold to all predictions
                for pred_data, gt_masks in zip(all_predictions, all_masks):
                    pred_labels = pred_data['pred_labels'].clone()
                    max_probs = pred_data['max_probs']
                    
                    # Apply confidence threshold
                    low_confidence_mask = max_probs < threshold
                    pred_labels = pred_labels.masked_fill(low_confidence_mask, 0)
                    
                    miou_metric.update(pred_labels, gt_masks)
                
                miou = miou_metric.compute()
                threshold_results[threshold] = miou
                
                logger.info(f"[{dataset_type.upper()}] Threshold {threshold:.2f}: mIoU = {miou*100:.4f}%")
                self.writer.add_scalar(f"val_th/{dataset_type}/mIoU_thresh_{threshold:.2f}", miou, epoch)
                
                # Check if mIoU decreased compared to previous threshold
                if previous_miou is not None and miou < previous_miou:
                    logger.info(f"⚠️  [{dataset_type.upper()}] mIoU decreased from {previous_miou*100:.4f}% to {miou*100:.4f}%. Stopping threshold search.")
                    break
                
                if miou > best_miou:
                    best_miou = miou
                    best_threshold = threshold
                
                previous_miou = miou  # Update previous mIoU for next iteration
            
            logger.info(f"{'='*60}")
            logger.info(f"✅ [{dataset_type.upper()}] Best Threshold: {best_threshold:.2f} with mIoU = {best_miou*100:.4f}%")
            logger.info(f"{'='*60}\n")
        else:
            # No threshold evaluation for Cityscapes and others
            best_miou = miou_metric.compute()
            best_threshold = None
            logger.info(f"\n{'='*60}")
            logger.info(f"Validation - {dataset_type.upper()} (Epoch {epoch+1})")
            logger.info(f"{'='*60}")
            logger.info(f"✅ [{dataset_type.upper()}] mIoU = {best_miou*100:.4f}%")
            logger.info(f"{'='*60}\n")
        
        # Use best threshold for visualization
        if save_visualizations:
            vis_count = 0
            if use_threshold:
                # Pascal VOC: apply threshold to predictions
                for pred_data, gt_masks in zip(all_predictions, all_masks):
                    if vis_count >= 10:
                        break
                        
                    if pred_data['images'] is not None:
                        pred_labels = pred_data['pred_labels'].clone()
                        max_probs = pred_data['max_probs']
                        low_confidence_mask = max_probs < best_threshold
                        pred_labels = pred_labels.masked_fill(low_confidence_mask, 0)
                        
                        self._save_validation_visualizations(
                            pred_data['images'], gt_masks, pred_labels, epoch, vis_count, dataset_type, dataset_config
                        )
                        vis_count += len(pred_data['images'])
            else:
                # Cityscapes and others: use stored predictions directly
                if hasattr(miou_metric, 'stored_images'):
                    for images, gt_masks, pred_labels in zip(miou_metric.stored_images, miou_metric.stored_masks, miou_metric.stored_predictions):
                        if vis_count >= 10:
                            break
                        self._save_validation_visualizations(
                            images, gt_masks, pred_labels, epoch, vis_count, dataset_type, dataset_config
                        )
                        vis_count += len(images)
        
        # Per-threshold and best_threshold logging moved to train() function
        # (val_ep, val_th, val_max categories)

        # === Multi-step ODE Inference Analysis ===
        # Run validation with 1..N ODE steps (using best_threshold found above).
        # Only applicable when use_text_flow=True; skipped otherwise.
        step_miou_results = {}
        if infer_steps_list and self.head.use_text_flow and self._val_clip_cache_tensor is not None:
            logger.info(f"\n{'='*60}")
            logger.info(f"Multi-step Inference Analysis - {dataset_type.upper()} (Epoch {epoch+1})")
            logger.info(f"Steps to evaluate: {infer_steps_list}")
            logger.info(f"Fixed threshold for BG: {best_threshold}")
            logger.info(f"{'='*60}")

            with torch.no_grad():
                for n_steps in tqdm(infer_steps_list, desc=f"Multi-step ({dataset_type})", leave=False):
                    # Encode text features using n_steps ODE integration steps
                    tf_s = l2norm(self.head.apply_text_flow_with_steps(
                        self._val_clip_cache_tensor, n_steps
                    ))

                    miou_s = MeanIoU(
                        num_classes=num_classes,
                        device=self.device,
                        ignore_indices=ignore_indices,
                        include_background=include_bg_in_miou
                    )

                    _sw_ode_si = 0
                    for batch in dataloader:
                        if use_slide_window:
                            # ── Sliding-window path for multi-step ODE ──────────
                            pil_images_s = batch.get('pil_images', [])
                            pil_masks_s  = batch.get('pil_masks',  [])

                            _tf_s = tf_s  # capture in closure
                            head_fn_s = lambda A: (
                                torch.einsum('bdhw,kd->bkhw',
                                             self.head.process_visual_features(A)[1],
                                             _tf_s) / self.head.tau
                            )

                            for _sw_j, (pil_img_s, pil_mask_s) in enumerate(zip(pil_images_s, pil_masks_s)):
                                _sw_ds_idx_s = _sw_ode_si + _sw_j
                                _sw_cf_s = slide_cache_dir / f"{_sw_ds_idx_s:06d}.npy" if slide_cache_dir is not None else None
                                if _sw_cf_s is not None and _sw_cf_s.exists():
                                    _sw_cached_s = np.load(str(_sw_cf_s), allow_pickle=True).item()
                                    logits_map_s = self._sliding_window_inference_from_cache(_sw_cached_s, head_fn_s)
                                else:
                                    logits_map_s = self._sliding_window_inference(
                                        pil_img_s, head_fn_s,
                                        crop_size=slide_crop_size,
                                        stride=slide_stride,
                                        max_long_side=slide_max_long_side,
                                    )  # [K, H_r, W_r]

                                mask_np_s = np.array(pil_mask_s, dtype=np.int64)
                                if reduce_zero_label:
                                    mask_np_s = _rzl_np(mask_np_s)
                                mask_single_s = torch.from_numpy(mask_np_s).unsqueeze(0)  # [1, H_o, W_o]

                                logits_up_s = F.interpolate(
                                    logits_map_s.unsqueeze(0),
                                    size=mask_single_s.shape[-2:],
                                    mode='bilinear', align_corners=False,
                                )
                                if _n_bg_channels > 1:
                                    _bg_max_s = logits_up_s[:, :_n_bg_channels].max(dim=1, keepdim=True)[0]
                                    logits_up_s = torch.cat([_bg_max_s, logits_up_s[:, _n_bg_channels:]], dim=1)
                                pred_probs_s = F.softmax(logits_up_s, dim=1)  # [1, K, H, W]

                                if use_pamr:
                                    img_t_s = (
                                        torch.from_numpy(np.array(pil_img_s, dtype=np.float32) / 255.0)
                                        .permute(2, 0, 1).unsqueeze(0).to(self.device)
                                    )
                                    _pH_s, _pW_s = pred_probs_s.shape[-2:]
                                    _pamr_chunk_s = max(1, min(30, int(2e9 / (48 * _pH_s * _pW_s * 4))))
                                    with torch.no_grad():
                                        for _pc_s in range(0, pred_probs_s.shape[1], _pamr_chunk_s):
                                            pred_probs_s[:, _pc_s:_pc_s + _pamr_chunk_s] = self.trainer.apply_pamr(
                                                img_t_s, pred_probs_s[:, _pc_s:_pc_s + _pamr_chunk_s]
                                            )

                                pred_labels_s = torch.argmax(pred_probs_s, dim=1).cpu()   # [1, H, W]
                                max_probs_s   = torch.max(pred_probs_s, dim=1)[0].cpu()   # [1, H, W]

                                if best_threshold is not None and best_threshold > 0.0:
                                    low_conf_s = max_probs_s < best_threshold
                                    pred_labels_s = pred_labels_s.masked_fill_(low_conf_s, 0)

                                miou_s.update(pred_labels_s, mask_single_s.cpu())
                            _sw_ode_si += len(pil_images_s)
                        else:
                            # ── Standard (whole-image) path ─────────────────────
                            dino_A_s = batch.get('dino_A')
                            if dino_A_s is not None:
                                A_s = dino_A_s.to(self.device)
                            else:
                                images_s = batch['image'].to(self.device)
                                A_s, _ = self.backbone.forward_grid(images_s)

                            masks_s = batch['mask']
                            if reduce_zero_label:
                                masks_s = _rzl_tensor(masks_s)
                            _, Z_map_s = self.head.process_visual_features(A_s)
                            logits_s = torch.einsum('bdhw,kd->bkhw', Z_map_s, tf_s) / self.head.tau
                            pred_masks_s = F.interpolate(logits_s, size=masks_s.shape[-2:], mode='bilinear', align_corners=False)
                            if _n_bg_channels > 1:
                                _bg_max_s = pred_masks_s[:, :_n_bg_channels].max(dim=1, keepdim=True)[0]
                                pred_masks_s = torch.cat([_bg_max_s, pred_masks_s[:, _n_bg_channels:]], dim=1)
                            pred_probs_s = F.softmax(pred_masks_s, dim=1)
                            pred_labels_s = torch.argmax(pred_masks_s, dim=1).cpu()
                            max_probs_s = torch.max(pred_probs_s, dim=1)[0].cpu()

                            if best_threshold is not None and best_threshold > 0.0:
                                low_conf_s = max_probs_s < best_threshold
                                pred_labels_s = pred_labels_s.masked_fill_(low_conf_s, 0)

                            miou_s.update(pred_labels_s, masks_s.cpu())

                    step_miou_s = miou_s.compute()
                    step_miou_results[n_steps] = step_miou_s
                    train_marker = " ← training" if n_steps == getattr(self.head, 'text_flow_steps', None) else ""
                    logger.info(f"  Step {n_steps:2d}: mIoU = {step_miou_s*100:.4f}%{train_marker}")

            logger.info(f"{'='*60}\n")

        # Log to text file (fancy format)
        with open(self.log_file, "a") as f:
            f.write("\n" + "="*80 + "\n")
            f.write(f"Validation ({dataset_type.upper()}) - Epoch [{epoch+1:3d}]\n")
            f.write("="*80 + "\n")
            if best_threshold is not None:
                f.write(f"mIoU: {best_miou*100:.4f}% (Best Threshold: {best_threshold:.2f})\n")
                f.write("\nThreshold Results:\n")
                for thresh in sorted(threshold_results.keys()):
                    marker = " ⭐" if thresh == best_threshold else ""
                    f.write(f"  - {thresh:.2f}: {threshold_results[thresh]*100:.4f}%{marker}\n")
            else:
                f.write(f"mIoU: {best_miou*100:.4f}%\n")
            if step_miou_results:
                f.write("\nMulti-step Inference (ODE steps vs mIoU):\n")
                for n_steps in sorted(step_miou_results.keys()):
                    train_marker = " ← train steps" if n_steps == getattr(self.head, 'text_flow_steps', None) else ""
                    f.write(f"  Steps {n_steps:2d}: {step_miou_results[n_steps]*100:.4f}%{train_marker}\n")
            f.write("="*80 + "\n\n")

        return {"mIoU": best_miou, "best_threshold": best_threshold, "threshold_results": threshold_results, "step_miou_results": step_miou_results}

    def _get_voc_palette(self):
        """Get Pascal VOC color palette."""
        # Pascal VOC 21-class color palette
        voc_palette = [
            [0, 0, 0],        # background
            [128, 0, 0],      # aeroplane
            [0, 128, 0],      # bicycle
            [128, 128, 0],    # bird
            [0, 0, 128],      # boat
            [128, 0, 128],    # bottle
            [0, 128, 128],    # bus
            [128, 128, 128],  # car
            [64, 0, 0],       # cat
            [192, 0, 0],      # chair
            [64, 128, 0],     # cow
            [192, 128, 0],    # diningtable
            [64, 0, 128],     # dog
            [192, 0, 128],    # horse
            [64, 128, 128],   # motorbike
            [192, 128, 128],  # person
            [0, 64, 0],       # pottedplant
            [128, 64, 0],     # sheep
            [0, 192, 0],      # sofa
            [128, 192, 0],    # train
            [0, 64, 128],     # tvmonitor
        ]
        return np.array(voc_palette) / 255.0  # Normalize to [0, 1]

    def _get_voc20_palette(self):
        """Get Pascal VOC20 color palette."""
        # Pascal VOC20 uses the same palette as VOC21
        voc20_palette = self._get_voc_palette()
        voc20_palette[0] = [1.0, 1.0, 1.0]  # Set Ignore to white for visualization
        return voc20_palette

    def _get_voc_context_palette(self):
        """Get Pascal Context color palette."""
        # Pascal Context 60-class color palette
        CONTEXT_PALETTE = [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
                 [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
                 [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
                 [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
                 [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
                 [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
                 [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
                 [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
                 [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
                 [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
                 [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
                 [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
                 [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
                 [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
                 [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255]]
        return np.array(CONTEXT_PALETTE) / 255.0  # Normalize to [0, 1]

    def _get_context59_palette(self):
        """Get Pascal Context59 color palette."""
        # Pascal Context59 uses the same palette as Context60, but sets Ignore to white
        context59_palette = self._get_voc_context_palette()
        context59_palette[0] = [1.0, 1.0, 1.0]  # Set Ignore to white for visualization
        return context59_palette

    def _get_coco_object_palette(self):
        """Get COCO Object color palette."""
        # COCO Object 81-class color palette
        OBJECT_PALETTE = [[0, 0, 0], [0, 192, 64], [0, 192, 64], [0, 64, 96], [128, 192, 192], [0, 64, 64], [0, 192, 224],
                 [0, 192, 192], [128, 192, 64], [0, 192, 96], [128, 192, 64], [128, 32, 192], [0, 0, 224], [0, 0, 64],
                 [0, 160, 192], [128, 0, 96], [128, 0, 192], [0, 32, 192], [128, 128, 224], [0, 0, 192],
                 [128, 160, 192],
                 [128, 128, 0], [128, 0, 32], [128, 32, 0], [128, 0, 128], [64, 128, 32], [0, 160, 0], [0, 0, 0],
                 [192, 128, 160], [0, 32, 0], [0, 128, 128], [64, 128, 160], [128, 160, 0], [0, 128, 0], [192, 128, 32],
                 [128, 96, 128], [0, 0, 128], [64, 0, 32], [0, 224, 128], [128, 0, 0], [192, 0, 160], [0, 96, 128],
                 [128, 128, 128], [64, 0, 160], [128, 224, 128], [128, 128, 64], [192, 0, 32],
                 [128, 96, 0], [128, 0, 192], [0, 128, 32], [64, 224, 0], [0, 0, 64], [128, 128, 160], [64, 96, 0],
                 [0, 128, 192], [0, 128, 160], [192, 224, 0], [0, 128, 64], [128, 128, 32], [192, 32, 128],
                 [0, 64, 192],
                 [0, 0, 32], [64, 160, 128], [128, 64, 64], [128, 0, 160], [64, 32, 128], [128, 192, 192], [0, 0, 160],
                 [192, 160, 128], [128, 192, 0], [128, 0, 96], [192, 32, 0], [128, 64, 128], [64, 128, 96],
                 [64, 160, 0],
                 [0, 64, 0], [192, 128, 224], [64, 32, 0], [0, 192, 128], [64, 128, 224], [192, 160, 0]]
        return np.array(OBJECT_PALETTE) / 255.0  # Normalize to [0, 1]
    
    def _get_coco_stuff_palette(self):
        """Get COCO Stuff color palette."""
        # COCO Stuff 171-class color palette
        STUFF_PALETTE = [[0, 192, 64], [0, 192, 64], [0, 64, 96], [128, 192, 192],
                 [0, 64, 64], [0, 192, 224], [0, 192, 192], [128, 192, 64],
                 [0, 192, 96], [128, 192, 64], [128, 32, 192], [0, 0, 224],
                 [0, 0, 64], [0, 160, 192], [128, 0, 96], [128, 0, 192],
                 [0, 32, 192], [128, 128, 224], [0, 0, 192], [128, 160, 192],
                 [128, 128, 0], [128, 0, 32], [128, 32, 0], [128, 0, 128],
                 [64, 128, 32], [0, 160, 0], [0, 0, 0], [192, 128, 160],
                 [0, 32, 0], [0, 128, 128], [64, 128, 160], [128, 160, 0],
                 [0, 128, 0], [192, 128, 32], [128, 96, 128], [0, 0, 128],
                 [64, 0, 32], [0, 224, 128], [128, 0, 0], [192, 0, 160],
                 [0, 96, 128], [128, 128, 128], [64, 0, 160], [128, 224, 128],
                 [128, 128, 64], [192, 0, 32], [128, 96, 0], [128, 0, 192],
                 [0, 128, 32], [64, 224, 0], [0, 0, 64], [128, 128, 160],
                 [64, 96, 0], [0, 128, 192], [0, 128, 160], [192, 224, 0],
                 [0, 128, 64], [128, 128, 32], [192, 32, 128], [0, 64, 192],
                 [0, 0, 32], [64, 160, 128], [128, 64, 64], [128, 0, 160],
                 [64, 32, 128], [128, 192, 192], [0, 0, 160], [192, 160, 128],
                 [128, 192, 0], [128, 0, 96], [192, 32, 0], [128, 64, 128],
                 [64, 128, 96], [64, 160, 0], [0, 64, 0], [192, 128, 224],
                 [64, 32, 0], [0, 192, 128], [64, 128, 224], [192, 160, 0],
                 [0, 192, 0], [192, 128, 96], [192, 96, 128], [0, 64, 128],
                 [64, 0, 96], [64, 224, 128], [128, 64, 0], [192, 0, 224],
                 [64, 96, 128], [128, 192, 128], [64, 0, 224], [192, 224, 128],
                 [128, 192, 64], [192, 0, 96], [192, 96, 0], [128, 64, 192],
                 [0, 128, 96], [0, 224, 0], [64, 64, 64], [128, 128, 224],
                 [0, 96, 0], [64, 192, 192], [0, 128, 224], [128, 224, 0],
                 [64, 192, 64], [128, 128, 96], [128, 32, 128], [64, 0, 192],
                 [0, 64, 96], [0, 160, 128], [192, 0, 64], [128, 64, 224],
                 [0, 32, 128], [192, 128, 192], [0, 64, 224], [128, 160, 128],
                 [192, 128, 0], [128, 64, 32], [128, 32, 64], [192, 0, 128],
                 [64, 192, 32], [0, 160, 64], [64, 0, 0], [192, 192, 160],
                 [0, 32, 64], [64, 128, 128], [64, 192, 160], [128, 160, 64],
                 [64, 128, 0], [192, 192, 32], [128, 96, 192], [64, 0, 128],
                 [64, 64, 32], [0, 224, 192], [192, 0, 0], [192, 64, 160],
                 [0, 96, 192], [192, 128, 128], [64, 64, 160], [128, 224, 192],
                 [192, 128, 64], [192, 64, 32], [128, 96, 64], [192, 0, 192],
                 [0, 192, 32], [64, 224, 64], [64, 0, 64], [128, 192, 160],
                 [64, 96, 64], [64, 128, 192], [0, 192, 160], [192, 224, 64],
                 [64, 128, 64], [128, 192, 32], [192, 32, 192], [64, 64, 192],
                 [0, 64, 32], [64, 160, 192], [192, 64, 64], [128, 64, 160],
                 [64, 32, 192], [192, 192, 192], [0, 64, 160], [192, 160, 192],
                 [192, 192, 0], [128, 64, 96], [192, 32, 64], [192, 64, 128],
                 [64, 192, 96], [64, 160, 64], [64, 64, 0]]
        return np.array(STUFF_PALETTE) / 255.0  # Normalize to [0, 1]
    
    def _get_cityscapes_palette(self):
        """Get Cityscapes color palette."""
        # Cityscapes 19-class standard palette (trainId 0-18)
        cityscapes_palette = np.array([
            [128,  64, 128],  # 0: road
            [244,  35, 232],  # 1: sidewalk
            [ 70,  70,  70],  # 2: building
            [102, 102, 156],  # 3: wall
            [190, 153, 153],  # 4: fence
            [153, 153, 153],  # 5: pole
            [250, 170,  30],  # 6: traffic light
            [220, 220,   0],  # 7: traffic sign
            [107, 142,  35],  # 8: vegetation
            [152, 251, 152],  # 9: terrain
            [ 70, 130, 180],  # 10: sky
            [220,  20,  60],  # 11: person
            [255,   0,   0],  # 12: rider
            [  0,   0, 142],  # 13: car
            [  0,   0,  70],  # 14: truck
            [  0,  60, 100],  # 15: bus
            [  0,  80, 100],  # 16: train
            [  0,   0, 230],  # 17: motorcycle
            [119,  11,  32],  # 18: bicycle
        ], dtype=np.uint8)
        return cityscapes_palette / 255.0  # Normalize to [0, 1]
    
    def _get_ade20k_palette(self):
        """Get ADE20K color palette."""
        # ADE20K uses a predefined color palette for 151 classes
        ade20k_palette = np.array([[0,0,0], [120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
                 [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
                 [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
                 [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
                 [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
                 [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
                 [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
                 [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
                 [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
                 [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
                 [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
                 [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
                 [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
                 [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
                 [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
                 [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
                 [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
                 [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
                 [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
                 [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
                 [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
                 [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
                 [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
                 [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
                 [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
                 [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
                 [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
                 [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
                 [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
                 [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
                 [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
                 [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
                 [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
                 [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
                 [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
                 [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
                 [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
                 [102, 255, 0], [92, 0, 255]], dtype=np.uint8)
        
        ade20k_palette[0] = [255, 255, 255]  # Set Ignore to white for visualization

        return ade20k_palette / 255.0  # Normalize to [0, 1]

    def _save_validation_visualizations(self, images, masks, pred_labels, epoch, sample_count, dataset_type, dataset_config=None):
        """Save validation visualizations for first few samples."""
        # Create visualization directory
        vis_dir = self.output_dir / "validation_visualizations" / dataset_type / f"epoch_{epoch:03d}"
        vis_dir.mkdir(parents=True, exist_ok=True)
        
        # Get class names and palette based on dataset type
        if dataset_type == "pascal_voc":
            class_names = [
                'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
                'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
            ]
            palette = self._get_voc_palette()
            num_classes = 21

        elif dataset_type == "voc20":
            class_names = [f'class_{i}' for i in range(21)]  # VOC20 has 21 classes (including ignore0)
            palette = self._get_voc20_palette()
            num_classes = 21                        

        elif dataset_type == "pascal_context":
            class_names = [f'class_{i}' for i in range(60)]  # Pascal Context has 60 classes
            palette = self._get_voc_context_palette()
            num_classes = 60
        
        elif dataset_type == "context59":
            class_names = [f'class_{i}' for i in range(60)]  # Pascal Context59 has 60 classes (including ignore0)
            palette = self._get_context59_palette()
            num_classes = 60

        elif dataset_type == "coco_stuff":
            class_names = [f'class_{i}' for i in range(171)]  # COCO-Stuff has 171 classes
            palette = self._get_coco_stuff_palette()
            num_classes = 171
        
        elif dataset_type == "coco_object":
            class_names = [f'class_{i}' for i in range(81)]  # COCO-Object has 81 classes
            palette = self._get_coco_object_palette()
            num_classes = 81

        elif dataset_type == "cityscapes":
            class_names = [
                'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
                'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
                'truck', 'bus', 'train', 'motorcycle', 'bicycle'
            ]
            palette = self._get_cityscapes_palette()
            num_classes = 19  # Cityscapes trainId range is 0-18

        elif dataset_type == "ade20k":
            # ADE20K has 151 classes
            # Use fallback class names for visualization
            class_names = [f'class_{i}' for i in range(151)]
            palette = self._get_ade20k_palette()
            num_classes = 151
            
        else:
            class_names = [f'class_{i}' for i in range(171)]  # Fallback for COCO-Stuff
            palette = None
            num_classes = None
        
        batch_size = images.shape[0]
        for i in range(min(batch_size, 10 - sample_count)):
            if sample_count + i >= 10:
                break
                
            # Denormalize image
            image = images[i].cpu()
            image = torch.clamp(image, 0, 1)
            
            # Get masks
            gt_mask = masks[i].cpu().numpy()
            pred_mask = pred_labels[i].cpu().numpy()
            
            # Convert 255 (ignore label) to 0 for visualization
            gt_mask = np.where(gt_mask == 255, 0, gt_mask)
            
            # Create figure
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            
            # Original image
            axes[0].imshow(image.permute(1, 2, 0))
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            # Apply palette to masks if available
            if palette is not None and num_classes is not None:
                # Clamp mask values to valid range
                gt_mask_clamped = np.clip(gt_mask, 0, num_classes - 1)
                pred_mask_clamped = np.clip(pred_mask, 0, num_classes - 1)
                
                # Convert masks to RGB using palette
                gt_mask_rgb = palette[gt_mask_clamped]
                pred_mask_rgb = palette[pred_mask_clamped]
                
                # Ground truth mask
                axes[1].imshow(gt_mask_rgb)
                axes[1].set_title('Ground Truth')
                axes[1].axis('off')
                
                # Prediction mask
                axes[2].imshow(pred_mask_rgb)
                axes[2].set_title('Prediction')
                axes[2].axis('off')
                
                # Overlay
                axes[3].imshow(image.permute(1, 2, 0))
                axes[3].imshow(pred_mask_rgb, alpha=0.6)
                axes[3].set_title('Overlay')
                axes[3].axis('off')
            else:
                # Fallback to tab20 colormap for non-Pascal VOC
                axes[1].imshow(gt_mask, cmap='tab20', vmin=0, vmax=20)
                axes[1].set_title('Ground Truth')
                axes[1].axis('off')
                
                axes[2].imshow(pred_mask, cmap='tab20', vmin=0, vmax=20)
                axes[2].set_title('Prediction')
                axes[2].axis('off')
                
                axes[3].imshow(image.permute(1, 2, 0))
                axes[3].imshow(pred_mask, cmap='tab20', vmin=0, vmax=20, alpha=0.6)
                axes[3].set_title('Overlay')
                axes[3].axis('off')
            
            plt.tight_layout()
            
            # Save visualization
            save_path = vis_dir / f"sample_{sample_count + i:03d}.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
        logger.info(f"Saved validation visualizations to {vis_dir}")

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Saves a training checkpoint.
        
        This function saves the full training state (model head, optimizer, scheduler, config, metrics)
        to a `.pth` file. It also provides options to save a "best_model.pth" if the current model
        achieves the best validation performance, and a separate head checkpoint for inference.

        Args:
            epoch (int): The current epoch number.
            metrics (Dict[str, float]): A dictionary containing current epoch's training and validation metrics.
            is_best (bool): A boolean flag indicating if the current model achieved the best validation loss.
                            Defaults to False.
        """
        checkpoint_dir = self.output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pth"
        
        # Include config in checkpoint
        ckpt_data = {
            'epoch': epoch,
            'model_state_dict': self.head.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'metrics': metrics
        }
        torch.save(ckpt_data, model_path)
        logger.info(f"Checkpoint saved to {model_path}")
        
        if is_best:
            best_model_path = checkpoint_dir / "best_model.pth"
            torch.save(ckpt_data, best_model_path)
            logger.info(f"Best model saved to {best_model_path}")
            
        # Also save the head model separately (for easier inference loading)
        head_path = checkpoint_dir / f"head_epoch_{epoch:03d}.pth"
        self.save_head_to_ckpt(self.head, head_path)
        
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Loads a training checkpoint to resume training.
        
        Loads the state dictionaries for the model head, optimizer, and scheduler.
        It also restores the starting epoch for training continuity.

        Args:
            checkpoint_path (str): The file path to the checkpoint to be loaded.

        Returns:
            int: The epoch number from which training should resume.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.head.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        logger.info(f"Resumed from checkpoint {checkpoint_path} (Epoch {start_epoch})")
        return start_epoch

    def train(self, train_dataloader: DataLoader, val_dataloader: Optional[DataLoader] = None, val_dataset_type: str = "coco_stuff", val_dataloaders: Optional[Dict] = None, processed_data_dir: Optional[str] = None, val_data_dirs: Optional[Dict[str, Optional[str]]] = None):

        self.val_clip_cache = {}
        self.val_data_dirs = val_data_dirs or {}

        # Load checkpoint if specified
        checkpoint_path = self.config["training"].get("resume_from_checkpoint")
        if checkpoint_path and checkpoint_path is not False:
            start_epoch = self.load_checkpoint(checkpoint_path)
        else:
            start_epoch = 0

        best_miou = 0.0
        best_epoch = -1
        best_epoch_metrics = {}  # Store {dataset_type: mIoU} from best epoch
        best_epoch_thresholds = {}  # Store {dataset_type: threshold} from best epoch

        # --- Pre-compute caption CLIP embeddings cache ---
        self.caption_clip_cache_tensor = None
        if processed_data_dir is not None:
            text_enc_cfg = self.config["text_encoder"]
            model_name = text_enc_cfg["model_name"].replace("/", "-")
            pretrained = text_enc_cfg["pretrained"]
            train_npy = "coco_stuff_train"
            cache_filename = f"caption_clip_cache_{train_npy}_{model_name}_{pretrained}.npy"
            cache_path = Path(processed_data_dir) / cache_filename

            if cache_path.exists():
                logger.info(f"Loading cached caption CLIP embeddings from {cache_path}")
                caption_clip_cache = np.load(cache_path)
            else:
                logger.info(f"Pre-computing caption CLIP embeddings for all training samples...")
                N = len(train_dataloader.dataset)
                caption_clip_cache = np.zeros((N, 768), dtype=np.float32)
                for i in tqdm(range(N), desc="Encoding captions"):
                    sample = train_dataloader.dataset[i]
                    caps = sample.get('captions', [])
                    if caps and len(caps) > 0:
                        with torch.no_grad():
                            emb = self.text_encoder.encode(caps, aggregate="mean")  # [1, 768]
                            caption_clip_cache[i] = emb.squeeze(0).cpu().numpy()
                np.save(cache_path, caption_clip_cache)
                valid_count = np.sum(np.linalg.norm(caption_clip_cache, axis=1) > 1e-6)
                logger.info(f"Saved caption CLIP cache to {cache_path} ({valid_count}/{N} valid)")

            self.caption_clip_cache_tensor = torch.from_numpy(caption_clip_cache).to(self.device)
            logger.info(f"Caption CLIP cache loaded: {self.caption_clip_cache_tensor.shape}")

        # --- Pre-compute DINOv3 backbone features cache (per-image files) ---
        # Skipped under --no_cache: the dataset then yields dino_A=None and the backbone
        # runs on the fly, which is slower but needs no extra disk.
        if processed_data_dir is not None and not self.no_cache:
            backbone_cfg = self.config["backbone"]
            backbone_id = backbone_cfg["model_id"].replace("/", "-")
            image_size = backbone_cfg["image_size"]
            dino_cache_dir = Path(processed_data_dir) / f"dino_feature_{backbone_id}_{image_size}" / "train"

            N = len(train_dataloader.dataset)
            existing_count = len(list(dino_cache_dir.glob("*.npy"))) if dino_cache_dir.exists() else 0

            if existing_count >= N:
                logger.info(f"DINOv3 per-image cache complete at {dino_cache_dir} ({existing_count} files)")
            else:
                dino_cache_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Creating DINOv3 per-image cache at {dino_cache_dir} ({existing_count}/{N} exist)...")

                processor = self.backbone.processor
                cache_batch_size = 256
                for start_idx in tqdm(range(0, N, cache_batch_size), desc="Caching DINOv3 features"):
                    end_idx = min(start_idx + cache_batch_size, N)

                    # Collect items that need caching
                    items_to_cache = []
                    for i in range(start_idx, end_idx):
                        raw_sample = train_dataloader.dataset.processed_data[i]
                        img_name = os.path.splitext(os.path.basename(str(raw_sample['image_path'])))[0]
                        cache_file = dino_cache_dir / f"{img_name}.npy"
                        if not cache_file.exists():
                            items_to_cache.append((i, raw_sample, img_name))

                    if not items_to_cache:
                        continue

                    pil_images = [Image.open(str(item[1]['image_path'])).convert('RGB') for item in items_to_cache]
                    processed = processor(images=pil_images, return_tensors="pt")
                    images_tensor = processed["pixel_values"].to(self.device)
                    with torch.no_grad():
                        A, cls_token = self.backbone.forward_grid(images_tensor)

                    A_np = A.cpu().numpy().astype(np.float16)
                    cls_np = cls_token.cpu().numpy().astype(np.float16)
                    for j, (_, _, img_name) in enumerate(items_to_cache):
                        np.save(str(dino_cache_dir / f"{img_name}.npy"), {'A': A_np[j], 'cls': cls_np[j]})

                logger.info(f"Saved DINOv3 per-image cache to {dino_cache_dir}")

            # Set cache dir on dataset so dataloader workers load per-image features
            train_dataloader.dataset.dino_cache_dir = str(dino_cache_dir)
            logger.info(f"DINOv3 per-image cache enabled: {dino_cache_dir}")
   
        num_epochs = self.config["training"]["num_epochs"]
        self.trainer.scheduler.T_max = num_epochs * len(train_dataloader)

        # Multi-step inference evaluation configuration
        infer_steps_list = list(self.config.get("flow", {}).get("infer_steps", range(1, 11)))
        # best_step_miou_by_dataset[dataset_type][n_steps] = best mIoU seen so far
        best_step_miou_by_dataset: Dict[str, Dict[int, float]] = {}

        for epoch in range(start_epoch, num_epochs):
            logger.info(f"--- Epoch {epoch+1}/{num_epochs} ---")
            
            train_metrics = self.train_epoch(train_dataloader, epoch)
            
            # Run validation on all specified datasets
            if val_dataloaders and (epoch + 1) % self.config['training']['eval_interval'] == 0:
                
                all_val_metrics = {}
                all_val_thresholds = {}
                for dataset_type, val_dl in val_dataloaders.items():
                    # Save visualizations for supported datasets
                    # Note: For COCO-Stuff, we don't save visualizations as it has 171 classes
                    save_vis = (dataset_type in ["pascal_voc", "voc20", "pascal_context", "context59", "coco_object", "coco_stuff", "cityscapes", "ade20k"])
                    val_metrics = self.validate(val_dl, epoch, dataset_type=dataset_type, save_visualizations=save_vis, infer_steps_list=infer_steps_list)
                    all_val_metrics[dataset_type] = val_metrics['mIoU']
                    all_val_thresholds[dataset_type] = val_metrics['best_threshold']

                    # Log per-epoch metrics: val_ep/{dataset_type}/mIoU
                    self.writer.add_scalar(f"val_ep/{dataset_type}/mIoU", val_metrics['mIoU']*100, epoch)

                    logger.info(f"📊 {dataset_type.upper()} Validation - mIoU: {val_metrics['mIoU']*100:.4f}%")

                    # Log multi-step mIoU: val_ep_{dataset_type}/step_{N}/mIoU
                    # and update val_max_{dataset_type}/step_{N}/mIoU
                    step_results = val_metrics.get('step_miou_results', {})
                    if step_results:
                        if dataset_type not in best_step_miou_by_dataset:
                            best_step_miou_by_dataset[dataset_type] = {}
                        for n_steps, step_miou in step_results.items():
                            self.writer.add_scalar(
                                f"val_ep_{dataset_type}/step_{n_steps}/mIoU", step_miou * 100, epoch
                            )
                            # Update best per step
                            prev_best = best_step_miou_by_dataset[dataset_type].get(n_steps, 0.0)
                            best_step_miou_by_dataset[dataset_type][n_steps] = max(step_miou, prev_best)
                            self.writer.add_scalar(
                                f"val_max_{dataset_type}/step_{n_steps}/mIoU",
                                best_step_miou_by_dataset[dataset_type][n_steps] * 100,
                                epoch
                            )
                
                # Calculate average mIoU across all validation datasets
                avg_miou = sum(all_val_metrics.values()) / len(all_val_metrics) if all_val_metrics else 0.0
                self.writer.add_scalar(f"val_ep/Avg/mIoU", avg_miou*100, epoch)
                logger.info(f"📊 Average Validation mIoU: {avg_miou*100:.4f}%")
                
                # Track best epoch based on average mIoU
                if avg_miou > best_miou:
                    best_miou = avg_miou
                    best_epoch = epoch + 1
                    best_epoch_metrics = all_val_metrics.copy()
                    best_epoch_thresholds = all_val_thresholds.copy()
                    logger.info(f"🏆 New best Average mIoU: {best_miou*100:.4f}% at epoch {best_epoch}")
                    self.save_checkpoint(epoch, train_metrics, is_best=True)
                else:
                    logger.info(f"📊 Current best Average mIoU: {best_miou*100:.4f}% at epoch {best_epoch}")
                
                # Log best epoch metrics: val_max/Avg/mIoU and val_max/{dataset_type}/mIoU
                self.writer.add_scalar(f"val_max/Avg/mIoU", best_miou*100, epoch)
                for dataset_type in all_val_metrics.keys():
                    if dataset_type in best_epoch_metrics:
                        self.writer.add_scalar(f"val_max/{dataset_type}/mIoU", best_epoch_metrics[dataset_type]*100, epoch)
                        if best_epoch_thresholds.get(dataset_type) is not None:
                            self.writer.add_scalar(f"val_max_th/{dataset_type}", best_epoch_thresholds[dataset_type], epoch)
                
            if (epoch + 1) % self.config['training']['save_interval'] == 0:
                self.save_checkpoint(epoch, train_metrics)
        
        # Final summary
        logger.info("="*80)
        logger.info(f"Training completed!")
        if best_epoch > 0:
            logger.info(f"🏆 Best mIoU: {best_miou*100:.4f}% achieved at epoch {best_epoch}")
        logger.info("="*80)
                

def load_config(config_path: str) -> Dict:
    """Loads a JSON configuration file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def set_seed(seed: int):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    """Main entry point for the DINOv3 OV-SS Training Pipeline (PASCAL VOC / Custom Dataset).
    
    Parses command-line arguments for configuration, dataset paths, output directory,
    and options for custom datasets. It loads the configuration, sets up the appropriate
    training and validation datasets and DataLoaders, and then initializes and runs
    the `TrainingPipeline`.
    """
    parser = argparse.ArgumentParser(description="Training script for DINOv3 Open-Vocabulary Segmentation on COCO-Stuff.")
    parser.add_argument("--config", type=str, default="configs/coco_stuff_config.json", help="Path to the configuration file.")
    parser.add_argument("--output_dir", type=str, default="outputs_coco_stuff", help="Directory to save outputs.")
    parser.add_argument("--processed_data_dir", type=str, required=True, help="Directory with processed COCO-Stuff .npy files.")
    parser.add_argument("--val_dataset", type=str, default="coco_stuff",
                        help="""Validation datasets to use. Use comma-separated for multiple.
Available protocols:
  BG Include: pascal_voc (VOC21), pascal_context (Context60), coco_object (81 classes)
  BG Exclude: voc20, context59, coco_stuff (171 classes), cityscapes (19), ade20k (150)
Example: pascal_voc,voc20,cityscapes,coco_stuff,ade20k""")
    parser.add_argument("--pascal_voc_data_dir", type=str, default=None,
                        help="Directory with processed Pascal VOC .npy files (required if pascal_voc or voc20 in val_dataset).")
    parser.add_argument("--pascal_context_data_dir", type=str, default=None,
                        help="Directory with processed Pascal Context .npy files (required if pascal_context or context59 in val_dataset).")
    parser.add_argument("--coco_object_data_dir", type=str, default=None,
                        help="Directory with processed COCO Object .npy files (required if coco_object in val_dataset).")
    parser.add_argument("--coco_stuff_data_dir", type=str, default=None,
                        help="Directory with processed COCO Stuff .npy files (required if coco_stuff in val_dataset).")
    parser.add_argument("--cityscapes_data_dir", type=str, default=None,
                        help="Directory with processed Cityscapes .npy files (required if cityscapes in val_dataset).")
    parser.add_argument("--ade20k_data_dir", type=str, default=None,
                        help="Directory with processed ADE20K .npy files (required if ade20k in val_dataset).")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to use (for debugging)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training.")
    parser.add_argument("--no_cache", action="store_true",
                        help="If set, do not write the DINOv3 dense-feature caches; compute backbone features on the fly instead. Much slower, but avoids hundreds of GB of disk (see the caching section of README.md).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    
    args = parser.parse_args()

    # Set random seed for reproducibility
    set_seed(args.seed)

    config = load_config(args.config)
    
    if args.resume:
        config["training"]["resume_from_checkpoint"] = args.resume

    # Setup dataset and dataloaders
    # Image sizes for preprocessing
    train_image_size = config["data"]["image_size"]  # 224
    val_image_size = config["data"].get("validation_image_size", 448)
    
    # Get processor from config - will be used in collate functions
    from model.components import DinoV3HFBackbone
    _temp_backbone = DinoV3HFBackbone(
        model_id=config["backbone"]["model_id"],
        device=config.get("device", "cuda"),
        image_size=train_image_size
    )
    _processor = _temp_backbone.processor
    del _temp_backbone  # Free backbone, only keep processor
    
    def collate_train(batch):
        """Custom collate function using DINOv3 processor for training."""
        pil_masks = [item['mask'] for item in batch]
        class_names_list = [item['class_names'] for item in batch]
        captions_list = [item.get('captions', []) for item in batch]
        dataset_indices = [item.get('dataset_idx', -1) for item in batch]

        # Check if cached DINOv3 features are available (from per-image .npy files)
        has_dino_cache = batch[0].get('dino_A') is not None

        if has_dino_cache:
            # Use cached features, skip image processing
            dino_A = torch.from_numpy(np.stack([item['dino_A'] for item in batch]))
            dino_cls = torch.from_numpy(np.stack([item['dino_cls'] for item in batch]))
            
            images = torch.empty(0)  # dummy, not used when cache available
        else:
            # Process images with DINOv3 processor
            pil_images = [item['image'] for item in batch]
            processed = _processor(images=pil_images, return_tensors="pt")
            images = processed["pixel_values"]  # [B, 3, H, W]
            dino_A = None
            dino_cls = None

        # Resize and convert masks to tensors
        masks = []
        for mask in pil_masks:
            mask = mask.resize((train_image_size, train_image_size), Image.NEAREST)
            mask_array = np.array(mask, dtype=np.int64)
            mask_tensor = torch.from_numpy(mask_array)  # [H, W]
            masks.append(mask_tensor)
        masks = torch.stack(masks)  # [B, H, W]

        return {
            'image': images,
            'mask': masks,
            'class_names': class_names_list,
            'captions': captions_list,
            'dataset_idx': dataset_indices,
            'dino_A': dino_A,
            'dino_cls': dino_cls,
        }
    
    def collate_val(batch):
        """Custom collate function using DINOv3 processor for validation."""
        pil_images_orig = [item['image'] for item in batch]   # original PIL images (for slide window)
        pil_masks_orig  = [item['mask']  for item in batch]   # original PIL masks  (for slide window)
        class_names_list = [item['class_names'] for item in batch]
        captions_list = [item.get('captions', []) for item in batch]
        dataset_indices = [item.get('dataset_idx', -1) for item in batch]

        # Check if cached DINOv3 features are available
        has_dino_cache = batch[0].get('dino_A') is not None

        if has_dino_cache:
            dino_A = torch.from_numpy(np.stack([item['dino_A'] for item in batch]))
            dino_cls = torch.from_numpy(np.stack([item['dino_cls'] for item in batch]))

            images = torch.stack([
                torch.from_numpy(np.array(img.resize((val_image_size, val_image_size), Image.BILINEAR), dtype=np.float32)).permute(2, 0, 1) / 255.0
                for img in pil_images_orig
            ])
        else:
            original_size = _processor.size
            _processor.size = {"height": val_image_size, "width": val_image_size}
            processed = _processor(images=pil_images_orig, return_tensors="pt")
            images = processed["pixel_values"]
            _processor.size = original_size
            dino_A = None
            dino_cls = None

        masks = []
        for mask in pil_masks_orig:
            mask_resized = mask.resize((val_image_size, val_image_size), Image.NEAREST)
            masks.append(torch.from_numpy(np.array(mask_resized, dtype=np.int64)))
        masks = torch.stack(masks)

        return {
            'image': images,
            'mask': masks,
            'pil_images': pil_images_orig,   # list of original PIL images (for slide window)
            'pil_masks':  pil_masks_orig,    # list of original PIL masks  (for slide window)
            'class_names': class_names_list,
            'captions': captions_list,
            'dataset_idx': dataset_indices,
            'dino_A': dino_A,
            'dino_cls': dino_cls,
        }

    # Compute DINOv3 per-image cache directory
    backbone_id = config["backbone"]["model_id"].replace("/", "-")
    train_image_size_cfg = config["backbone"]["image_size"]
    # Under --no_cache the dataset gets no cache dir at all, so a cache left over from an
    # earlier run is not silently reused and features are always computed on the fly.
    dino_cache_dir = (
        None if args.no_cache
        else os.path.join(args.processed_data_dir, f"dino_feature_{backbone_id}_{train_image_size_cfg}", "train")
    )

    # Setup datasets
    train_data_path = os.path.join(args.processed_data_dir, "coco_stuff_train.npy")
    train_dataset = COCOStuffDataset(processed_data_path=train_data_path, transform=None, mask_transform=None, max_samples=args.max_samples, dino_cache_dir=dino_cache_dir)
    
    # Parse validation datasets
    val_datasets_list = [d.strip() for d in args.val_dataset.split(',')]
    
    # Use smaller batch size for validation to avoid memory issues
    val_batch_size = config.get("validation", {}).get("batch_size", config["training"]["batch_size"] // 4)
    val_batch_size = max(1, val_batch_size)  # At least 1
    logger.info(f"Validation batch size: {val_batch_size} (Training batch size: {config['training']['batch_size']})")
    
    # Setup validation datasets
    # 8 Evaluation Protocols:
    # BG Include: pascal_voc (VOC21), pascal_context (Context60), coco_object (81 classes)
    # BG Exclude: voc20, context59, coco_stuff (171 classes), cityscapes (19), ade20k (150)
    val_dataloaders = {}
    num_classes = 171  # Default to COCO-Stuff (171 classes, BG excluded)

    for val_dataset_type in val_datasets_list:
        # === BG INCLUDE PROTOCOLS ===
        # Compute val dino_cache_dir for this dataset type
        _val_data_dir_map = {
            'pascal_voc': args.pascal_voc_data_dir, 'voc20': args.pascal_voc_data_dir,
            'pascal_context': args.pascal_context_data_dir, 'context59': args.pascal_context_data_dir,
            'coco_object': args.coco_object_data_dir, 'coco_stuff': args.processed_data_dir,
            'cityscapes': args.cityscapes_data_dir, 'ade20k': args.ade20k_data_dir,
        }
        _vdd = _val_data_dir_map.get(val_dataset_type)
        val_dino_cache = (
            os.path.join(_vdd, f"dino_feature_{backbone_id}_{val_image_size}", "val")
            if _vdd and not args.no_cache else None
        )

        if val_dataset_type == "pascal_voc":  # VOC21 - 21 classes with BG
            if args.pascal_voc_data_dir is None:
                raise ValueError("--pascal_voc_data_dir must be provided when using Pascal VOC validation dataset")
            val_data_path = os.path.join(args.pascal_voc_data_dir, "pascal_voc_val.npy")
            val_dataset = PascalVOCDataset(processed_data_path=val_data_path, transform=None, mask_transform=None, max_samples=args.max_samples, dino_cache_dir=val_dino_cache)
            num_classes = 21
            logger.info(f"[VOC21] Pascal VOC validation: 21 classes (BG include), {val_image_size}x{val_image_size}")

        elif val_dataset_type == "pascal_context":  # Context60 - 60 classes with BG
            if args.pascal_context_data_dir is None:
                raise ValueError("--pascal_context_data_dir must be provided when using Pascal Context validation dataset")
            val_data_path = os.path.join(args.pascal_context_data_dir, "pascal_context_val.npy")
            val_dataset = PascalContextDataset(processed_data_path=val_data_path, transform=None, mask_transform=None, max_samples=args.max_samples, dino_cache_dir=val_dino_cache)
            num_classes = 60
            logger.info(f"[Context60] Pascal Context validation: 60 classes (BG include), {val_image_size}x{val_image_size}")

        elif val_dataset_type == "coco_object":  # COCO Object - 81 classes with BG
            if args.coco_object_data_dir is None:
                raise ValueError("--coco_object_data_dir must be provided when using COCO Object validation dataset")
            val_data_path = os.path.join(args.coco_object_data_dir, "coco_object_val.npy")
            val_dataset = COCOObjectDataset(processed_data_path=val_data_path, transform=None, mask_transform=None, max_samples=args.max_samples, bg_mode='include', dino_cache_dir=val_dino_cache)
            num_classes = 81
            logger.info(f"[COCO Object] COCO Object validation: 81 classes (BG include), {val_image_size}x{val_image_size}")

        # === BG EXCLUDE PROTOCOLS ===
        elif val_dataset_type == "voc20":  # VOC20 - 20 classes, BG excluded from mIoU
            if args.pascal_voc_data_dir is None:
                raise ValueError("--pascal_voc_data_dir must be provided when using VOC20 validation dataset")
            val_data_path = os.path.join(args.pascal_voc_data_dir, "pascal_voc_val.npy")
            val_dataset = PascalVOCDataset(processed_data_path=val_data_path, transform=None, mask_transform=None, max_samples=args.max_samples, dino_cache_dir=val_dino_cache)
            num_classes = 21
            logger.info(f"[VOC20] Pascal VOC validation: 20 classes (BG exclude from mIoU), {val_image_size}x{val_image_size}")

        elif val_dataset_type == "context59":  # Context59 - 59 classes, BG excluded from mIoU
            if args.pascal_context_data_dir is None:
                raise ValueError("--pascal_context_data_dir must be provided when using Context59 validation dataset")
            val_data_path = os.path.join(args.pascal_context_data_dir, "pascal_context_val.npy")
            val_dataset = PascalContextDataset(processed_data_path=val_data_path, transform=None, mask_transform=None, max_samples=args.max_samples, dino_cache_dir=val_dino_cache)
            num_classes = 60
            logger.info(f"[Context59] Pascal Context validation: 59 classes (BG exclude from mIoU), {val_image_size}x{val_image_size}")

        elif val_dataset_type == "coco_stuff":  # COCO-Stuff 171 - NO BG
            val_data_path = os.path.join(args.processed_data_dir, "coco_stuff_val.npy")
            val_dataset = COCOStuffDataset(processed_data_path=val_data_path, transform=None, mask_transform=None, max_samples=args.max_samples, dino_cache_dir=val_dino_cache)
            num_classes = 171
            logger.info(f"[COCO-Stuff] COCO-Stuff validation: 171 classes (NO BG), {val_image_size}x{val_image_size}")

        elif val_dataset_type == "cityscapes":  # Cityscapes - 19 classes, NO BG
            if args.cityscapes_data_dir is None:
                raise ValueError("--cityscapes_data_dir must be provided when using Cityscapes validation dataset")
            val_dataset = CityscapesDataset(processed_data_path=args.cityscapes_data_dir, split='val', transform=None, dino_cache_dir=val_dino_cache)
            num_classes = 19
            logger.info(f"[Cityscapes] Cityscapes validation: 19 classes (NO BG), {val_image_size}x{val_image_size}")

        elif val_dataset_type == "ade20k":  # ADE20K - 150 classes, NO BG
            if args.ade20k_data_dir is None:
                raise ValueError("--ade20k_data_dir must be provided when using ADE20K validation dataset")
            val_data_path = os.path.join(args.ade20k_data_dir, "ade20k_val.npy")
            val_dataset = ADE20KDataset(processed_data_path=val_data_path, transform=None, mask_transform=None, max_samples=args.max_samples, dino_cache_dir=val_dino_cache)
            num_classes = 151
            logger.info(f"[ADE20K] ADE20K validation: 151 classes (but ignore background 0 in mIoU), {val_image_size}x{val_image_size}")

        else:
            logger.warning(f"Unknown dataset type: {val_dataset_type}, skipping...")
            continue
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=config["data"]["num_workers"],
            pin_memory=False,   # True
            collate_fn=collate_val
        )
        val_dataloaders[val_dataset_type] = val_dataloader

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"],
        pin_memory=False,   # True
        collate_fn=collate_train
    )

    # Build mapping: dataset_type -> data directory for validation cache
    val_data_dirs = {
        'pascal_voc': args.pascal_voc_data_dir,
        'voc20': args.pascal_voc_data_dir,
        'pascal_context': args.pascal_context_data_dir,
        'context59': args.pascal_context_data_dir,
        'coco_object': args.coco_object_data_dir,
        'coco_stuff': args.processed_data_dir,
        'cityscapes': args.cityscapes_data_dir,
        'ade20k': args.ade20k_data_dir,
    }

    # Add timestamp prefix to output_dir (MMDD_HHMMSS format)
    if '/' in args.output_dir:
        base_dir = os.path.dirname(args.output_dir)
        dir_name = os.path.basename(args.output_dir)
        timestamp = datetime.now().strftime("%m%d_%H%M%S")
        output_dir_with_timestamp = os.path.join(base_dir, f"{timestamp}_{dir_name}")
    else:
        timestamp = datetime.now().strftime("%m%d_%H%M%S")
        output_dir_with_timestamp = f"{timestamp}_{args.output_dir}"

    # Initialize and run pipeline
    pipeline = TrainingPipeline(config, output_dir=output_dir_with_timestamp, num_classes=num_classes,
                                no_cache=args.no_cache)
    pipeline.train(train_dataloader, val_dataloaders=val_dataloaders, processed_data_dir=args.processed_data_dir, val_data_dirs=val_data_dirs)

if __name__ == "__main__":
    main()
