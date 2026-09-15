# DINOv3 Open-Vocabulary Semantic Segmentation Evaluation Pipeline (Flow Version)
#
# Eval-only pipeline. Loads a checkpoint and runs validation on any combination
# of the 8 evaluation protocols without any training loop.
#
# Key Features:
# - Sliding-window inference with pre-computed DINOv3 crop feature cache.
# - PAMR (Pixel Adaptive Mask Refinement) post-processing.
# - Multi-step ODE inference analysis (infer_steps from config).
# - Supports all 8 OV-SS evaluation protocols (VOC21/20, Context60/59,
#   COCO Object, COCO-Stuff, Cityscapes, ADE20K).
# - Checkpoint loaded via --checkpoint CLI argument.

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
logger = logging.getLogger("DINOde-EVAL")

# Model implementation file, snapshotted into <output_dir>/code_backup for reproducibility.
MODEL_FILE = "dinode.py"

class EvalPipeline:
    """DINOv3 OV-SS Evaluation Pipeline (Flow Version).

    Loads a pre-trained checkpoint and evaluates on any of the 8 OV-SS
    evaluation protocols.  No training loop; inference only.
    """
    def __init__(self, config: Dict, output_dir: str = "outputs_eval", num_classes: int = 182,
                 no_cache: bool = False):
        """
        Initializes the EvalPipeline.

        Creates output directories, saves the config, determines the device,
        and initializes model components (backbone, text encoder, head).

        Args:
            config (Dict): Configuration dictionary (same format as training config).
            output_dir (str): Base output directory for logs and visualizations.
            num_classes (int): Number of classes (used for logging only).
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
        self.log_file = self.output_dir / "eval_log.txt"

        # Write header
        with open(self.log_file, "w") as f:
            f.write("="*80 + "\n")
            f.write("Evaluation Log\n")
            f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")

        logger.info(f"Evaluation logs will be saved to: {self.log_file}")
        
        self._init_models()
        
        param_counts = count_parameters(self.head)
        logger.info(f"Model Head Parameters: {param_counts}")
        self.writer.add_text("model", f"Total Parameters: {param_counts['total_params']}\nTrainable Parameters: {param_counts['trainable_params']}")
        
        logger.info(f"Eval pipeline initialized. Device: {self.device}")

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
        
        self.head = TextCondHead(
            in_channels=head_config["in_channels"],
            tau=head_config["tau"],
            use_text_flow=head_config.get("use_text_flow", True),
            use_cls_flow=head_config.get("use_cls_flow", True),
            use_cls_mlp=head_config.get("use_cls_mlp", False),
            text_flow_steps=flow_config.get("steps", 10),
            text_flow_depth=flow_config.get("depth", 4),
            text_flow_dt=flow_config.get("dt"),  # Use config dt if provided
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
                # Per-step threshold search: {step_num: [candidates]}. Missing keys use default [0.0..0.9].
                "step_thresholds": {
                    1:  [0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14], # 0.09
                    2:  [0.09, 0.10, 0.11, 0.12, 0.13, 0.14], # 0.10
                    3:  [0.10, 0.11, 0.12, 0.13, 0.14], # 0.11
                    4:  [0.11, 0.12, 0.13, 0.14], # 0.12
                    5:  [0.12, 0.13, 0.14, 0.15, 0.16], # 0.13
                    6:  [0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.21, 0.22, 0.23, 0.24], # 0.16
                    7:  [0.18, 0.19, 0.20, 0.21, 0.22, 0.23, 0.24], # 0.19
                    8:  [0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26], # 0.21
                    9:  [0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28], # 0.23
                    10: [0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29], # 0.23
                },
                "has_background": True,
                "include_background_in_miou": True,
                "reduce_zero_label": False,
                "requires_label_mapping": False,
            },
            "pascal_context": {  # Context60 - 60 classes with background
                "num_classes": 60,
                "use_threshold": True,
                "thresholds": [0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14],
                "step_thresholds": {
                    1:  [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09],   # 0.03
                    2:  [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09],
                    3:  [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09],
                    4:  [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09],
                    5:  [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09],
                    6:  [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09],
                    7:  [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09],
                    8:  [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09],
                    9:  [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09],
                    10: [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09],
                },
                "has_background": True,
                "include_background_in_miou": True,
                "reduce_zero_label": False,
                "requires_label_mapping": False,
            },
            "coco_object": {  # COCO Object - 81 classes with background
                "num_classes": 81,
                "use_threshold": True,
                "thresholds": [0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14],
                "step_thresholds": {
                    1:  [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09],
                    2:  [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09],
                    3:  [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09],
                    4:  [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09],
                    5:  [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09],
                    6:  [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09],
                    7:  [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09],
                    8:  [0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13],
                    9:  [0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14],
                    10: [0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14],
                },
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
                "step_thresholds": {},  # BG-Exclude: threshold not used; code skips search
                "has_background": False,
                "include_background_in_miou": True,  # All 20 classes are FG
                "reduce_zero_label": True,            # GT 0→255(ignore), FG labels shift -1
                "requires_label_mapping": False,
            },
            "context59": {  # Context59 - 59 FG classes only (BG excluded via reduce_zero_label)
                "num_classes": 59,  # 59 FG classes only; GT 0→255 by reduce_zero_label
                "use_threshold": False,
                "thresholds": [0.0001],
                "step_thresholds": {},  # BG-Exclude: threshold not used; code skips search
                "has_background": False,
                "include_background_in_miou": True,  # All 59 classes are FG
                "reduce_zero_label": True,            # GT 0→255(ignore), FG labels shift -1
                "requires_label_mapping": False,
            },
            "coco_stuff": {  # COCO-Stuff 171 - 171 classes, NO background
                "num_classes": 171,  # 171 stuff classes (BG excluded)
                "use_threshold": False,
                "thresholds": [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09],
                "step_thresholds": {},  # BG-Exclude: threshold not used; code skips search
                "has_background": False,
                "include_background_in_miou": True,  # All classes included (no BG exists)
                "reduce_zero_label": False,
                "requires_label_mapping": False,
            },
            "cityscapes": {  # Cityscapes - 19 classes, NO background
                "num_classes": 19,
                "use_threshold": False,
                "thresholds": [0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14],
                "step_thresholds": {},  # BG-Exclude: threshold not used; code skips search
                "has_background": False,
                "include_background_in_miou": True,  # All classes included (no BG exists)
                "reduce_zero_label": False,
                "requires_label_mapping": False,
            },
            "ade20k": {  # ADE20K - 150 FG classes only (BG excluded via reduce_zero_label)
                "num_classes": 150,  # 150 FG classes only; GT 0→255 by reduce_zero_label
                "use_threshold": False,
                "thresholds": [0.0001],
                "step_thresholds": {},  # BG-Exclude: threshold not used; code skips search
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
            elif dataset_type == "voc20":  # VOC20 - BG exclude: 20 FG classes only (no 'background')
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
            elif dataset_type == "context59":  # Context59 - BG exclude: 59 FG classes only (no 'background')
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
            elif dataset_type == "ade20k":  # ADE20K - BG exclude: 150 FG classes only (no 'background')
                # 150 FG classes - background excluded; GT 0→255 via reduce_zero_label
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
            # Skipped under --no_cache: the dataset then yields dino_A=None and the backbone
            # runs on the fly, which is slower but needs no extra disk.
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

                # Class-name embeddings are recomputed for every evaluation instead of being
                # read from disk. They are cheap - a few hundred vectors - and recomputing
                # keeps a stale cache from outliving a change to the class list or to the
                # prompt templates in build_prompts().
                logger.info(f"Encoding class-name embeddings for {dataset_type}")
                with torch.no_grad():
                    T_clip = torch.stack([
                        self.text_encoder.encode(build_prompts(n, dataset_type), aggregate="mean").squeeze(0)
                        for n in _flat_class_names
                    ], dim=0)  # [M+N, 768] where M=_n_bg_channels, N=n_fg_classes
                self._val_clip_cache_tensor = T_clip

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

                        mask_np     = np.array(pil_mask, dtype=np.int64)
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
        # Each step independently searches for its best BG threshold.
        # step_thresholds: {step_num: [candidates]} from DATASET_INFO.
        # Missing step keys fall back to _default_step_thresholds.
        # Only applicable when use_text_flow=True; skipped otherwise.
        _step_thresholds_map = dataset_config.get("step_thresholds", {})
        _default_step_thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        step_miou_results = {}
        if infer_steps_list and self.head.use_text_flow and self._val_clip_cache_tensor is not None:
            logger.info(f"\n{'='*60}")
            logger.info(f"Multi-step Inference Analysis - {dataset_type.upper()} (Epoch {epoch+1})")
            logger.info(f"Steps to evaluate: {infer_steps_list}")
            if use_threshold:
                logger.info(f"Per-step threshold search enabled (step_thresholds map configured in DATASET_INFO)")
            else:
                logger.info(f"Per-step threshold search: N/A (BG-Exclude protocol)")
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

                    # Collect raw predictions for threshold search after full pass
                    all_step_preds_s = []  # list of {'pred_labels': tensor, 'max_probs': tensor}
                    all_step_masks_s = []  # list of GT tensors

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

                                mask_np_s   = np.array(pil_mask_s, dtype=np.int64)
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

                                # Collect (no threshold applied yet)
                                all_step_preds_s.append({'pred_labels': pred_labels_s, 'max_probs': max_probs_s})
                                all_step_masks_s.append(mask_single_s.cpu())
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

                            # Collect (no threshold applied yet)
                            all_step_preds_s.append({'pred_labels': pred_labels_s, 'max_probs': max_probs_s})
                            all_step_masks_s.append(masks_s.cpu())

                    # ── Per-step threshold search ─────────────────────────────
                    if use_threshold:
                        # Thresholds for this specific step number; fall back to default if not configured
                        thresholds_for_step = _step_thresholds_map.get(n_steps, _default_step_thresholds)
                        best_step_th   = None
                        best_step_miou = 0.0
                        prev_step_miou = None
                        for sth in thresholds_for_step:
                            miou_s.reset()
                            for pd_s, gm_s in zip(all_step_preds_s, all_step_masks_s):
                                pl_s = pd_s['pred_labels'].clone()
                                pl_s = pl_s.masked_fill(pd_s['max_probs'] < sth, 0)
                                miou_s.update(pl_s, gm_s)
                            sth_miou = miou_s.compute()
                            logger.info(f"  [Step {n_steps:2d}] [{dataset_type.upper()}] Threshold {sth:.2f}: mIoU = {sth_miou*100:.4f}%")
                            if sth_miou > best_step_miou:
                                best_step_miou = sth_miou
                                best_step_th   = sth
                            if prev_step_miou is not None and sth_miou < prev_step_miou:
                                logger.info(f"  [Step {n_steps:2d}] mIoU decreased ({prev_step_miou*100:.4f}% → {sth_miou*100:.4f}%), stopping threshold search.")
                                break  # early stop: mIoU started decreasing
                            prev_step_miou = sth_miou
                    else:
                        # BG-Exclude: no threshold search — direct compute
                        for pd_s, gm_s in zip(all_step_preds_s, all_step_masks_s):
                            miou_s.update(pd_s['pred_labels'], gm_s)
                        best_step_miou = miou_s.compute()
                        best_step_th   = None

                    step_miou_results[n_steps] = {"miou": best_step_miou, "best_threshold": best_step_th}
                    train_marker = " ← training" if n_steps == getattr(self.head, 'text_flow_steps', None) else ""
                    th_str = f" (best_th={best_step_th:.2f})" if best_step_th is not None else ""
                    logger.info(f"  Step {n_steps:2d}: mIoU = {best_step_miou*100:.4f}%{th_str}{train_marker}")

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
                f.write("\nMulti-step Inference (ODE steps vs mIoU, per-step best threshold):\n")
                for n_steps in sorted(step_miou_results.keys()):
                    train_marker = " ← train steps" if n_steps == getattr(self.head, 'text_flow_steps', None) else ""
                    sr = step_miou_results[n_steps]
                    th_str = f" (best_th={sr['best_threshold']:.2f})" if sr['best_threshold'] is not None else ""
                    f.write(f"  Steps {n_steps:2d}: {sr['miou']*100:.4f}%{th_str}{train_marker}\n")
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

    def load_checkpoint_for_eval(self, checkpoint_path: str) -> int:
        """Load model weights from a checkpoint for evaluation.

        Only loads ``model_state_dict``; optimizer / scheduler states are ignored.

        Args:
            checkpoint_path: Path to the ``.pth`` checkpoint file.

        Returns:
            The epoch number stored in the checkpoint (0 if not present).
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.head.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 0)
        logger.info(f"Loaded checkpoint: {checkpoint_path} (epoch {epoch})")
        return epoch

    def evaluate(
        self,
        val_dataloaders: Dict[str, DataLoader],
        val_data_dirs: Optional[Dict[str, Optional[str]]] = None,
        infer_steps_list: Optional[List[int]] = None,
        epoch: int = 0,
    ) -> Dict[str, Dict]:
        """Run evaluation on all provided validation dataloaders.

        Args:
            val_dataloaders: Mapping dataset_type -> DataLoader.
            val_data_dirs:   Mapping dataset_type -> data directory (for cache).
            infer_steps_list: ODE step counts to evaluate (multi-step analysis).
            epoch:           Epoch number to use for logging / visualization dirs.

        Returns:
            Mapping dataset_type -> metrics dict (as returned by ``validate``).
        """
        self.val_data_dirs = val_data_dirs or {}
        all_metrics: Dict[str, Dict] = {}

        for dataset_type, val_dl in val_dataloaders.items():
            save_vis = dataset_type in [
                "pascal_voc", "voc20", "pascal_context", "context59",
                "coco_object", "coco_stuff", "cityscapes", "ade20k",
            ]
            val_metrics = self.validate(
                val_dl,
                epoch=epoch,
                dataset_type=dataset_type,
                save_visualizations=save_vis,
                infer_steps_list=infer_steps_list,
            )
            all_metrics[dataset_type] = val_metrics
            self.writer.add_scalar(f"val_ep/{dataset_type}/mIoU", val_metrics['mIoU'] * 100, epoch)
            logger.info(f"[EVAL] {dataset_type.upper()} mIoU: {val_metrics['mIoU']*100:.4f}%")

            # Log per-step mIoU
            for n_steps, step_miou in val_metrics.get('step_miou_results', {}).items():
                _step_miou_val = step_miou['miou'] if isinstance(step_miou, dict) else step_miou
                self.writer.add_scalar(
                    f"val_ep_{dataset_type}/step_{n_steps}/mIoU", _step_miou_val * 100, epoch
                )

        # Summary
        avg_miou = (
            sum(m['mIoU'] for m in all_metrics.values()) / len(all_metrics)
            if all_metrics else 0.0
        )
        self.writer.add_scalar("val_ep/Avg/mIoU", avg_miou * 100, epoch)

        logger.info("=" * 80)
        logger.info("Evaluation complete!")
        logger.info(f"Average mIoU: {avg_miou*100:.4f}%")
        for dt, metrics in all_metrics.items():
            logger.info(f"  {dt.upper()}: {metrics['mIoU']*100:.4f}%")
        logger.info("=" * 80)

        return all_metrics
                

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
    """Entry point for the eval-only DINOv3 OV-SS pipeline.

    Loads a checkpoint and runs validation on the requested dataset protocols.
    No training loop is performed.
    """
    parser = argparse.ArgumentParser(
        description="Eval-only script for DINOv3 Open-Vocabulary Segmentation."
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the JSON configuration file.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the .pth checkpoint file to evaluate.")
    parser.add_argument("--output_dir", type=str, default="outputs_eval",
                        help="Directory to save evaluation outputs.")
    parser.add_argument("--val_dataset", type=str, default="coco_stuff",
                        help="""Comma-separated list of evaluation protocols.
Available protocols:
  BG Include: pascal_voc (VOC21), pascal_context (Context60), coco_object (81)
  BG Exclude: voc20, context59, coco_stuff (171), cityscapes (19), ade20k (150)
Example: pascal_voc,voc20,pascal_context,context59,coco_object,coco_stuff,cityscapes,ade20k""")
    parser.add_argument("--pascal_voc_data_dir", type=str, default=None,
                        help="Processed Pascal VOC directory (needed for pascal_voc / voc20).")
    parser.add_argument("--pascal_context_data_dir", type=str, default=None,
                        help="Processed Pascal Context directory (needed for pascal_context / context59).")
    parser.add_argument("--coco_object_data_dir", type=str, default=None,
                        help="Processed COCO Object directory (needed for coco_object).")
    parser.add_argument("--coco_stuff_data_dir", type=str, default=None,
                        help="Processed COCO-Stuff directory (needed for coco_stuff).")
    parser.add_argument("--cityscapes_data_dir", type=str, default=None,
                        help="Processed Cityscapes directory (needed for cityscapes).")
    parser.add_argument("--ade20k_data_dir", type=str, default=None,
                        help="Processed ADE20K directory (needed for ade20k).")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Cap dataset size (for quick debugging).")
    parser.add_argument("--no_cache", action="store_true",
                        help="If set, do not write the DINOv3 dense-feature caches; compute backbone features on the fly instead. Much slower, but avoids hundreds of GB of disk (see the caching section of README.md).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")

    args = parser.parse_args()
    set_seed(args.seed)

    config = load_config(args.config)

    val_image_size = config["data"].get("validation_image_size", 448)
    backbone_id    = config["backbone"]["model_id"].replace("/", "-")

    # ── Build val collate function ──────────────────────────────────────────
    from model.components import DinoV3HFBackbone as _BB
    _temp_bb = _BB(
        model_id=config["backbone"]["model_id"],
        device=config.get("device", "cuda"),
        image_size=config["backbone"]["image_size"],
    )
    _processor = _temp_bb.processor
    del _temp_bb

    def collate_val(batch):
        pil_images_orig = [item['image'] for item in batch]
        pil_masks_orig  = [item['mask']  for item in batch]
        class_names_list = [item['class_names'] for item in batch]
        captions_list    = [item.get('captions', []) for item in batch]
        dataset_indices  = [item.get('dataset_idx', -1) for item in batch]

        has_dino_cache = batch[0].get('dino_A') is not None
        if has_dino_cache:
            dino_A   = torch.from_numpy(np.stack([item['dino_A']   for item in batch]))
            dino_cls = torch.from_numpy(np.stack([item['dino_cls'] for item in batch]))
            images = torch.stack([
                torch.from_numpy(
                    np.array(img.resize((val_image_size, val_image_size), Image.BILINEAR),
                             dtype=np.float32)
                ).permute(2, 0, 1) / 255.0
                for img in pil_images_orig
            ])
        else:
            orig_size = _processor.size
            _processor.size = {"height": val_image_size, "width": val_image_size}
            processed = _processor(images=pil_images_orig, return_tensors="pt")
            images = processed["pixel_values"]
            _processor.size = orig_size
            dino_A = dino_cls = None

        masks = torch.stack([
            torch.from_numpy(
                np.array(m.resize((val_image_size, val_image_size), Image.NEAREST), dtype=np.int64)
            )
            for m in pil_masks_orig
        ])

        return {
            'image':       images,
            'mask':        masks,
            'pil_images':  pil_images_orig,
            'pil_masks':   pil_masks_orig,
            'class_names': class_names_list,
            'captions':    captions_list,
            'dataset_idx': dataset_indices,
            'dino_A':      dino_A,
            'dino_cls':    dino_cls,
        }

    # ── Build val dataloaders ───────────────────────────────────────────────
    val_batch_size = config.get("validation", {}).get("batch_size", 16)
    val_batch_size = max(1, val_batch_size)
    logger.info(f"Validation batch size: {val_batch_size}")

    val_datasets_list = [d.strip() for d in args.val_dataset.split(',')]
    val_dataloaders: Dict[str, DataLoader] = {}
    num_classes = 171  # default

    _val_data_dir_map = {
        'pascal_voc':      args.pascal_voc_data_dir,
        'voc20':           args.pascal_voc_data_dir,
        'pascal_context':  args.pascal_context_data_dir,
        'context59':       args.pascal_context_data_dir,
        'coco_object':     args.coco_object_data_dir,
        'coco_stuff':      args.coco_stuff_data_dir,
        'cityscapes':      args.cityscapes_data_dir,
        'ade20k':          args.ade20k_data_dir,
    }

    for val_dataset_type in val_datasets_list:
        _vdd = _val_data_dir_map.get(val_dataset_type)
        # Under --no_cache the dataset gets no cache dir at all, so a cache left over from an
        # earlier run is not silently reused and features are always computed on the fly.
        val_dino_cache = (
            os.path.join(_vdd, f"dino_feature_{backbone_id}_{val_image_size}", "val")
            if _vdd and not args.no_cache else None
        )

        if val_dataset_type == "pascal_voc":
            if args.pascal_voc_data_dir is None:
                raise ValueError("--pascal_voc_data_dir required for pascal_voc")
            val_dataset = PascalVOCDataset(
                processed_data_path=os.path.join(args.pascal_voc_data_dir, "pascal_voc_val.npy"),
                transform=None, mask_transform=None, max_samples=args.max_samples,
                dino_cache_dir=val_dino_cache)
            num_classes = 21
            logger.info(f"[VOC21] 21 classes (BG include), {val_image_size}x{val_image_size}")

        elif val_dataset_type == "voc20":
            if args.pascal_voc_data_dir is None:
                raise ValueError("--pascal_voc_data_dir required for voc20")
            val_dataset = PascalVOCDataset(
                processed_data_path=os.path.join(args.pascal_voc_data_dir, "pascal_voc_val.npy"),
                transform=None, mask_transform=None, max_samples=args.max_samples,
                dino_cache_dir=val_dino_cache)
            num_classes = 21
            logger.info(f"[VOC20] 20 classes (BG exclude), {val_image_size}x{val_image_size}")

        elif val_dataset_type == "pascal_context":
            if args.pascal_context_data_dir is None:
                raise ValueError("--pascal_context_data_dir required for pascal_context")
            val_dataset = PascalContextDataset(
                processed_data_path=os.path.join(args.pascal_context_data_dir, "pascal_context_val.npy"),
                transform=None, mask_transform=None, max_samples=args.max_samples,
                dino_cache_dir=val_dino_cache)
            num_classes = 60
            logger.info(f"[Context60] 60 classes (BG include), {val_image_size}x{val_image_size}")

        elif val_dataset_type == "context59":
            if args.pascal_context_data_dir is None:
                raise ValueError("--pascal_context_data_dir required for context59")
            val_dataset = PascalContextDataset(
                processed_data_path=os.path.join(args.pascal_context_data_dir, "pascal_context_val.npy"),
                transform=None, mask_transform=None, max_samples=args.max_samples,
                dino_cache_dir=val_dino_cache)
            num_classes = 60
            logger.info(f"[Context59] 59 classes (BG exclude), {val_image_size}x{val_image_size}")

        elif val_dataset_type == "coco_object":
            if args.coco_object_data_dir is None:
                raise ValueError("--coco_object_data_dir required for coco_object")
            val_dataset = COCOObjectDataset(
                processed_data_path=os.path.join(args.coco_object_data_dir, "coco_object_val.npy"),
                transform=None, mask_transform=None, max_samples=args.max_samples,
                bg_mode='include', dino_cache_dir=val_dino_cache)
            num_classes = 81
            logger.info(f"[COCO Object] 81 classes (BG include), {val_image_size}x{val_image_size}")

        elif val_dataset_type == "coco_stuff":
            if args.coco_stuff_data_dir is None:
                raise ValueError("--coco_stuff_data_dir required for coco_stuff")
            val_dataset = COCOStuffDataset(
                processed_data_path=os.path.join(args.coco_stuff_data_dir, "coco_stuff_val.npy"),
                transform=None, mask_transform=None, max_samples=args.max_samples,
                dino_cache_dir=val_dino_cache)
            num_classes = 171
            logger.info(f"[COCO-Stuff] 171 classes (NO BG), {val_image_size}x{val_image_size}")

        elif val_dataset_type == "cityscapes":
            if args.cityscapes_data_dir is None:
                raise ValueError("--cityscapes_data_dir required for cityscapes")
            val_dataset = CityscapesDataset(
                processed_data_path=args.cityscapes_data_dir, split='val',
                transform=None, dino_cache_dir=val_dino_cache)
            num_classes = 19
            logger.info(f"[Cityscapes] 19 classes (NO BG), {val_image_size}x{val_image_size}")

        elif val_dataset_type == "ade20k":
            if args.ade20k_data_dir is None:
                raise ValueError("--ade20k_data_dir required for ade20k")
            val_dataset = ADE20KDataset(
                processed_data_path=os.path.join(args.ade20k_data_dir, "ade20k_val.npy"),
                transform=None, mask_transform=None, max_samples=args.max_samples,
                dino_cache_dir=val_dino_cache)
            num_classes = 151
            logger.info(f"[ADE20K] 151 classes (BG exclude from mIoU), {val_image_size}x{val_image_size}")

        else:
            logger.warning(f"Unknown dataset type: {val_dataset_type}, skipping...")
            continue

        val_dataloaders[val_dataset_type] = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=config["data"]["num_workers"],
            pin_memory=False,
            collate_fn=collate_val,
        )

    # ── data-dir map for cache creation inside validate() ──────────────────
    val_data_dirs = {k: v for k, v in _val_data_dir_map.items()}

    # ── Output directory with timestamp ────────────────────────────────────
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    if '/' in args.output_dir:
        base_dir = os.path.dirname(args.output_dir)
        dir_name = os.path.basename(args.output_dir)
        output_dir_ts = os.path.join(base_dir, f"{timestamp}_{dir_name}")
    else:
        output_dir_ts = f"{timestamp}_{args.output_dir}"

    # ── Init pipeline, load checkpoint, evaluate ───────────────────────────
    pipeline = EvalPipeline(config, output_dir=output_dir_ts, num_classes=num_classes,
                            no_cache=args.no_cache)
    epoch = pipeline.load_checkpoint_for_eval(args.checkpoint)

    # Inference-only component ablations.
    # The full model is constructed and loaded first, so the official
    # checkpoint remains strictly compatible.
    ablate_text_flow = (
        os.environ.get(
            "DINODE_ABLATE_TEXT_FLOW",
            "0",
        )
        == "1"
    )
    ablate_cls_flow = (
        os.environ.get(
            "DINODE_ABLATE_CLS_FLOW",
            "0",
        )
        == "1"
    )

    if ablate_text_flow:
        def _text_init_only(T_clip):
            return torch.nn.functional.normalize(
                pipeline.head.text_flow_init(T_clip),
                dim=-1,
            )

        pipeline.head.apply_text_flow = _text_init_only
        logger.info(
            "[Ablation] Text ODE flow bypassed; "
            "using text_flow_init only"
        )

    if ablate_cls_flow:
        pipeline.head.use_cls_flow = False
        pipeline.head.use_cls_mlp = False
        logger.info(
            "[Ablation] CLS ODE flow bypassed; "
            "using cls_flow_init only"
        )

    infer_steps_list = list(config.get("flow", {}).get("infer_steps", range(1, 11)))

    pipeline.evaluate(
        val_dataloaders=val_dataloaders,
        val_data_dirs=val_data_dirs,
        infer_steps_list=infer_steps_list,
        epoch=epoch,
    )


if __name__ == "__main__":
    main()
