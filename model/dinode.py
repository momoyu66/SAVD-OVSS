"""
model/dinode.py

This module implements the core DINOv3-based Open-Vocabulary Semantic Segmentation model
with optional time-conditioned ODE-based flow for text feature alignment.

Key Components:
- TextCondHead: Text-conditioned segmentation head with optional ODE flow
- FlowTrainer: Training orchestrator with contrastive and rectified flow losses
- Time-conditioned ODE: Hypersphere-aware flow for text embedding transformation

The architecture uses DINOv3 for visual features, CLIP for text features, and optionally
learns a continuous mapping from text to visual space using time-conditioned ODE flow.
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler

import math

from .components import (
    l2norm,
    l2norm_hw,
    build_prompts,
    CLIPTextEncoder,
    DinoV3HFBackbone,
    logger
)


# ==================== Time Embedding ====================

def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Sinusoidal (Fourier) time embedding for better time representation.
    
    Args:
        t (torch.Tensor): Time values, shape [K] or scalar, values in [0, 1].
        dim (int): Output embedding dimension.
    
    Returns:
        torch.Tensor: Sinusoidal embedding, shape [K, dim].
    """
    t = t.view(-1)  # [K]
    half_dim = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half_dim, device=t.device, dtype=t.dtype) / half_dim
    )
    args = t[:, None] * freqs[None, :]  # [K, half_dim]
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [K, dim]
    return embedding


# ==================== Time-Conditioned Flow Network ====================

# ==================== Residual MLP Baseline ====================

class ResidualMLPBaseline(nn.Module):
    """
    Residual MLP baseline for comparison with ODE flow.
    
    Uses depth=4, hidden_dim=1024 to match FlowNet parameter count approximately.
    """
    def __init__(self, dim: int = 1024, hidden_dim: int = 1024, depth: int = 4):
        """
        Args:
            dim (int): Input/output dimension (1024 for DINOv3).
            hidden_dim (int): Hidden layer dimension. Defaults to 1024.
            depth (int): Number of residual blocks. Defaults to 4.
        """
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        
        # Residual MLP blocks
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, dim),
                # nn.LayerNorm(dim)
            ) for _ in range(depth)
        ])
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Applies residual MLP transformation.
        
        Args:
            z (torch.Tensor): Input embeddings, shape [K, dim].
        
        Returns:
            torch.Tensor: Transformed embeddings, shape [K, dim].
        """
        h = z
        for block in self.blocks:
            # Residual connection
            h = h + block(h)
        return h


class TimeConditionedTextFlowNet(nn.Module):
    """
    Time-conditioned MLP for text flow.
    
    Uses FiLM (Feature-wise Linear Modulation) to condition on time.
    """
    def __init__(self, dim: int = 1024, hidden_dim: int = 512, time_dim: int = 256, 
                 depth: int = 2):
        """
        Args:
            dim (int): Input/output dimension (1024 for DINOv3).
            hidden_dim (int): Hidden layer dimension.
            time_dim (int): Time embedding dimension.
            depth (int): Number of residual blocks.
        """
        super().__init__()
        self.dim = dim
        self.time_dim = time_dim
        
        # Time embedding MLP: sinusoidal → learned
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # FiLM parameters for each block (scale & shift)
        self.film_layers = nn.ModuleList([
            nn.Linear(time_dim, dim * 2) for _ in range(depth)
        ])
        
        # MLP blocks
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, dim)
            ) for _ in range(depth)
        ])
        
        # LayerNorm for stability
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(dim) for _ in range(depth)
        ])
        
        # Final velocity prediction layer (zero-initialized for stability)
        self.tail = nn.Linear(dim, dim, bias=False)
        nn.init.zeros_(self.tail.weight)
    
    def forward(self, z: torch.Tensor, t) -> torch.Tensor:
        """
        Predict velocity field V(z, t).
        
        Args:
            z (torch.Tensor): Input text embeddings, shape [K, dim].
            t: Time value(s). Can be scalar or [K] tensor, in range [0, 1].
        
        Returns:
            torch.Tensor: Predicted velocity, shape [K, dim].
        """
        K = z.shape[0]
        
        # Handle scalar or batch time
        if isinstance(t, (int, float)):
            t = torch.full((K,), t, device=z.device, dtype=z.dtype)
        elif torch.is_tensor(t) and t.dim() == 0:
            t = t.unsqueeze(0).expand(K)
        
        # Sinusoidal time embedding → MLP
        t_sin = sinusoidal_time_embedding(t, self.time_dim)  # [K, time_dim]
        t_emb = self.time_mlp(t_sin)  # [K, time_dim]
        
        # Pass through blocks with FiLM conditioning
        h = z
        for block, film, layer_norm in zip(
            self.blocks, self.film_layers, self.layer_norms
        ):
            # Get FiLM parameters
            film_params = film(t_emb)  # [K, dim*2]
            scale, shift = film_params.chunk(2, dim=-1)  # each [K, dim]
            
            # Apply MLP block with FiLM
            h_block = block(h)  # [K, dim]
            h_film = (1 + scale) * h_block + shift  # [K, dim]
            
            # Apply FiLM output with residual connection
            h = h + h_film  # Residual connection for FiLM
            
            # LayerNorm for stability
            h = layer_norm(h)
        
        # Final velocity prediction
        v = self.tail(h)
        return v


# ==================== Text Projection Wrapper ====================

class TextFlowProjection:
    """
    Wrapper class for ODE-based text projection.
    
    This class provides a callable interface compatible with the original text_proj,
    but uses ODE-based flow transformation instead of simple MLP.
    
    Note: This is NOT a nn.Module to avoid circular reference with TextCondHead.
    """
    def __init__(self, head):
        self.head = head
    
    def __call__(self, T_clip: torch.Tensor) -> torch.Tensor:
        """
        Applies ODE-based flow to transform CLIP text embeddings.
        
        Args:
            T_clip (torch.Tensor): CLIP text embeddings, shape `[K, 768]`.
        
        Returns:
            torch.Tensor: Transformed text embeddings, shape `[K, 1024]`.
        """
        return self.head.apply_text_flow(T_clip)
    
    def parameters(self):
        """Returns parameters from text_flow_init, text_flow_net (if enabled), or text_proj_mlp (if disabled)."""
        for param in self.head.text_flow_init.parameters():
            yield param
        if self.head.use_text_flow:
            for param in self.head.text_flow_net.parameters():
                yield param
        else:
            # If flow is disabled, include text_proj_mlp parameters
            for param in self.head.text_proj_mlp.parameters():
                yield param
    
    @property
    def weight(self):
        """For backward compatibility with device access."""
        return self.head.text_flow_init.weight


# ==================== Text-Conditioned Segmentation Head ====================

class TextCondHead(nn.Module):
    """
    Text-conditioned segmentation head for DINOv3 Open-Vocabulary Semantic Segmentation.

    This head processes DINOv3 visual features (1024 channels) conditioned on CLIP text features (768 channels).
    It optionally uses time-conditioned ODE flow to transform text embeddings to visual space.
    """
    def __init__(self, in_channels: int, tau: float = 0.07,
                 use_text_flow: bool = True,
                 text_flow_steps: int = 8,
                 text_flow_depth: int = 4,
                 text_flow_dt: Optional[float] = None,
                 use_cls_flow: bool = True,
                 use_cls_mlp: bool = False,
                 cls_flow_steps: Optional[int] = None,
                 cls_flow_depth: Optional[int] = None,
                 cls_flow_dt: Optional[float] = None,
                 topk: int = 20,
                 debug_interval: int = 100):
        """
        Initializes the TextCondHead.

        Args:
            in_channels (int): Number of input channels for image features. Must be 1024 for DINOv3.
            tau (float): Temperature parameter for logit scaling. Defaults to 0.07.
            use_text_flow (bool): Whether to use ODE-based flow for text transformation. Defaults to True.
            text_flow_steps (int): Number of ODE integration steps. Defaults to 8.
            text_flow_depth (int): Depth of flow network (number of residual blocks). Defaults to 4.
            text_flow_dt (Optional[float]): Time step size for ODE integration. If None, computed as 1.0/steps. Defaults to None.
            use_cls_flow (bool): Whether to use ODE-based flow for CLS token. Defaults to True.
            use_cls_mlp (bool): When use_cls_flow=False, whether to apply ResidualMLPBaseline after cls_flow_init. Defaults to False.
            cls_flow_steps (Optional[int]): ODE steps for CLS flow. If None, uses text_flow_steps.
            cls_flow_depth (Optional[int]): Depth of CLS flow network. If None, uses text_flow_depth.
            cls_flow_dt (Optional[float]): Time step for CLS flow. If None, computed from cls_flow_steps.
            topk (int): Number of top and bottom values to select in min-max k pooling. Defaults to 20.
            debug_interval (int): Interval for debug logging. Defaults to 100.
        """
        super().__init__()
        assert in_channels == 1024, "This head assumes DINOv3 large features (1024 channels)."
        self.in_channels = in_channels
        self.topk = topk
        self.use_text_flow = use_text_flow
        self.use_cls_flow = use_cls_flow
        self.use_cls_mlp = use_cls_mlp

        # Text feature projection: 768D CLIP → 1024D visual space
        self.text_flow_init = nn.Linear(768, in_channels, bias=False)

        # cls_flow_init is always created and trained regardless of use_cls_flow
        self.cls_flow_init = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.GELU(),
            nn.Linear(in_channels, in_channels),
        )
        
        # Time-conditioned text flow network (only if use_text_flow=True)
        if use_text_flow:
            self.text_flow_net = TimeConditionedTextFlowNet(
                dim=in_channels,
                hidden_dim=512,
                time_dim=256,
                depth=text_flow_depth
            )
            self.text_flow_steps = int(text_flow_steps)
            self.text_flow_depth = int(text_flow_depth)
            # Use config dt if provided, otherwise compute from steps
            if text_flow_dt is not None:
                self.text_flow_dt = float(text_flow_dt)
            else:
                self.text_flow_dt = float(1.0 / text_flow_steps)
            
            # Debug counter for time-conditioning analysis
            self._debug_counter = 0
            self._debug_interval = debug_interval
        else:
            # Residual MLP baseline (depth=4, hidden_dim=1024) for fair comparison
            self.text_proj_mlp = ResidualMLPBaseline(
                dim=in_channels,
                hidden_dim=1024,
                depth=4
            )
        
        # Create text_proj wrapper
        self.text_proj = TextFlowProjection(self)

        # self.cls_proj = nn.Sequential(
        #     nn.Linear(in_channels, in_channels),
        #     nn.GELU(),
        #     nn.Linear(in_channels, in_channels),
        # )

        # CLS token ODE flow (separate network from text_flow_net)
        if use_cls_flow:
            # _cls_steps = 10
            _cls_steps = int(text_flow_steps)
            _cls_depth = 4
            self.cls_flow_net = TimeConditionedTextFlowNet(
                dim=in_channels,
                hidden_dim=512,
                time_dim=256,
                depth=_cls_depth
            )
            self.cls_flow_steps = _cls_steps
            self.cls_flow_dt = float(1.0 / self.cls_flow_steps)
        elif use_cls_mlp:
            # When use_cls_flow=False and use_cls_mlp=True: apply ResidualMLP after cls_flow_init
            self.cls_proj_mlp = ResidualMLPBaseline(
                dim=in_channels,
                hidden_dim=1024,
                depth=4
            )

        # Logit scale parameter
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / max(tau, 1e-6))))

    @property
    def tau(self) -> float:
        """Returns the temperature (tau) value for logit scaling."""
        return float(1.0 / torch.exp(self.logit_scale))
    
    def apply_text_flow(self, T_clip: torch.Tensor) -> torch.Tensor:
        """
        Applies time-conditioned ODE-based flow to transform CLIP text embeddings (768D) to visual space (1024D).
        
        If use_text_flow=False, uses simple MLP projection instead.
        
        Args:
            T_clip (torch.Tensor): CLIP text embeddings, shape `[K, 768]`.
        
        Returns:
            torch.Tensor: Transformed text embeddings in visual space, shape `[K, 1024]`.
        """
        # Initial projection: 768D → 1024D
        Z = self.text_flow_init(T_clip)  # [K, 1024]
        Z = l2norm(Z)  # Normalize initial projection

        if not self.use_text_flow:
            # Residual MLP baseline projection
            Z = self.text_proj_mlp(Z)
            Z = l2norm(Z)
            return Z

        # === Debug: Check time-conditioning of flow ===
        if self.training:
            self._debug_counter += 1
            if self._debug_counter % self._debug_interval == 0:
                with torch.no_grad():
                    Z_test = Z[:1].clone()  # First sample
                    V_t0 = self.text_flow_net(Z_test, 0.0)
                    V_t05 = self.text_flow_net(Z_test, 0.5)
                    V_t1 = self.text_flow_net(Z_test, 1.0)
                    # Tangent projection at Z_test (same for all t at this point)
                    V_t0_proj = V_t0 - (V_t0 * Z_test).sum(dim=-1, keepdim=True) * Z_test
                    V_t05_proj = V_t05 - (V_t05 * Z_test).sum(dim=-1, keepdim=True) * Z_test
                    V_t1_proj = V_t1 - (V_t1 * Z_test).sum(dim=-1, keepdim=True) * Z_test
                    # Compare projected velocities
                    diff_0_05 = (V_t0_proj - V_t05_proj).norm().item()
                    diff_05_1 = (V_t05_proj - V_t1_proj).norm().item()
                    cos_sim_0_05 = F.cosine_similarity(V_t0_proj, V_t05_proj, dim=-1).mean().item()
                    cos_sim_0_1 = F.cosine_similarity(V_t0_proj, V_t1_proj, dim=-1).mean().item()
                    # V_proj stats (min, max, mean) per time
                    def _stats(v):
                        return v.min().item(), v.max().item(), v.mean().item(), v.norm().item()
                    m0, x0, a0, n0 = _stats(V_t0_proj)
                    m05, x05, a05, n05 = _stats(V_t05_proj)
                    m1, x1, a1, n1 = _stats(V_t1_proj)
                    logger.info(f"[Flow t-check #{self._debug_counter}] V_proj: "
                          f"||V_proj(0)-V_proj(0.5)||={diff_0_05:.4f}, ||V_proj(0.5)-V_proj(1)||={diff_05_1:.4f}, "
                          f"cos(V_proj_0,V_proj_0.5)={cos_sim_0_05:.4f}, cos(V_proj_0,V_proj_1)={cos_sim_0_1:.4f}")
                    logger.info(f"[Flow t-check #{self._debug_counter}] V_proj stats: "
                          f"t=0   min={m0:.4f} max={x0:.4f} mean={a0:.4f} norm={n0:.4f} | "
                          f"t=0.5 min={m05:.4f} max={x05:.4f} mean={a05:.4f} norm={n05:.4f} | "
                          f"t=1   min={m1:.4f} max={x1:.4f} mean={a1:.4f} norm={n1:.4f}")
        # === End debug ===

        # Time-conditioned ODE integration on the hypersphere: dZ/dt = V(Z, t)
        for step in range(self.text_flow_steps):
            t_val = step * self.text_flow_dt

            # 1. Predict the velocity field at the current point and time
            V = self.text_flow_net(Z, t_val)

            # 2. Tangent projection: project V onto the tangent plane at Z, so the
            #    update direction stays on the sphere's tangent bundle
            V = V - (V * Z).sum(dim=-1, keepdim=True) * Z

            # 3. Euler step followed by renormalization (retraction onto the sphere).
            #    The exact geodesic exponential map was also tried and performed on par:
            #      Z = Z * cos(|V| dt) + (V / |V|) * sin(|V| dt)
            Z = Z + self.text_flow_dt * V
        Z = l2norm(Z)
    
        return Z  # [K, 1024]

    def apply_text_flow_backward(self, z_end: torch.Tensor) -> torch.Tensor:
        """
        Integrates the text ODE backward in time from z_end (t=1) to z0 (t=0).
        Used for consistency loss: backward(forward(z0)) should approximate z0.

        Args:
            z_end (torch.Tensor): End point of text flow (t=1), shape [K, 1024], unit norm.

        Returns:
            torch.Tensor: Reconstructed z0, shape [K, 1024].
        """
        if not self.use_text_flow:
            return z_end
        Z = z_end
        for step in range(self.text_flow_steps):
            t_val = 1.0 - (step + 1) * self.text_flow_dt
            V = self.text_flow_net(Z, t_val)
            V = V - (V * Z).sum(dim=-1, keepdim=True) * Z
            Z = Z - self.text_flow_dt * V
        Z = l2norm(Z)
        return Z

    def apply_text_flow_with_steps(self, T_clip: torch.Tensor, num_steps: int) -> torch.Tensor:
        """
        Applies text ODE flow with a custom number of integration steps.

        Identical to apply_text_flow but uses num_steps instead of self.text_flow_steps,
        allowing inference-time analysis of step count vs. performance.

        Args:
            T_clip (torch.Tensor): CLIP text embeddings, shape [K, 768].
            num_steps (int): Number of Euler ODE integration steps.

        Returns:
            torch.Tensor: Transformed text embeddings on the sphere, shape [K, 1024].
        """
        Z = self.text_flow_init(T_clip)
        Z = l2norm(Z)

        if not self.use_text_flow:
            Z = self.text_proj_mlp(Z)
            Z = l2norm(Z)
            return Z

        # By default retain the official truncated-flow
        # analysis. When DINODE_RESCALED_STEPS=1,
        # every solver configuration integrates from
        # t=0 to t=1, providing a fair compute/accuracy
        # Euler baseline.
        import os
        if os.environ.get(
            "DINODE_RESCALED_STEPS",
            "0",
        ) == "1":
            dt = 1.0 / float(num_steps)
        else:
            dt = self.text_flow_dt

        for step in range(num_steps):
            t_val = step * dt
            V = self.text_flow_net(Z, t_val)
            V = V - (V * Z).sum(dim=-1, keepdim=True) * Z
            Z = Z + dt * V
        Z = l2norm(Z)
        return Z

    def apply_cls_flow(self, cls_token_raw: torch.Tensor) -> torch.Tensor:
        """
        Applies cls_flow_init to CLS token (always), then optionally:
          - use_cls_flow=True : integrates with cls_flow_net (ODE)
          - use_cls_flow=False, use_cls_mlp=True : applies cls_proj_mlp (ResidualMLP)
          - use_cls_flow=False, use_cls_mlp=False : returns cls_flow_init output only

        Args:
            cls_token_raw (torch.Tensor): Raw CLS token from backbone, shape [B, 1024].

        Returns:
            torch.Tensor: Transformed CLS token on sphere, shape [B, 1024].
        """
        # cls_flow_init is always applied and trained
        Z = l2norm(self.cls_flow_init(cls_token_raw))

        if self.use_cls_flow:
            for step in range(self.cls_flow_steps):
                t_val = step * self.cls_flow_dt
                V = self.cls_flow_net(Z, t_val)
                V = V - (V * Z).sum(dim=-1, keepdim=True) * Z
                Z = Z + self.cls_flow_dt * V
                
                # Z = l2norm(Z)
                
        elif self.use_cls_mlp:
            Z = self.cls_proj_mlp(Z)
            Z = l2norm(Z)
        # else: use_cls_flow=False and use_cls_mlp=False → return cls_flow_init output as-is
        return Z

    def process_visual_features(self, A: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Processes visual features.

        Args:
            A (torch.Tensor): Input visual features from the backbone, shape `[B, D, H, W]`.

        Returns:
            Tuple containing:
                - A_processed (torch.Tensor): Visual features (same as input).
                - Z_map (torch.Tensor): Normalized visual features.
        """
        Z_map = l2norm_hw(A)  # Normalize vision features
        return A, Z_map

    def forward(self, A: torch.Tensor, text_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Performs the full forward pass, including visual processing and dense logit calculation.

        Args:
            A (torch.Tensor): Input visual features from the backbone, shape `[B, D, H, W]`.
            text_features (torch.Tensor): Projected and normalized text features, shape `[K, D]`.

        Returns:
            Tuple containing:
                - logits (torch.Tensor): Dense similarity logits, shape `[B, K, H, W]`.
                - A_processed (torch.Tensor): Visual features (same as input).
                - Z_map (torch.Tensor): Normalized visual features.
        """
        A_processed, Z_map = self.process_visual_features(A)

        # Dense Logit Calculation
        logits = torch.einsum('bdhw,kd->bkhw', Z_map, text_features) / self.tau

        return logits, A_processed, Z_map


# ==================== Spatial Pooling Functions ====================

def min_max_k_pooling(x: torch.Tensor, k: int = 1) -> torch.Tensor:
    """
    Min-Max K pooling (per-channel): top-k and bottom-k values per channel.
    
    Args:
        x (torch.Tensor): Input tensor, shape [B, D, H, W]
        k (int): Number of top and bottom values to select. Defaults to 1.
    
    Returns:
        torch.Tensor: Pooled features, shape [B, D]
    """
    B, D, H, W = x.shape
    
    # Flatten spatial dimensions: [B, D, H*W]
    x_flat = x.view(B, D, -1)
    
    # Get top-k maximum values: [B, D, k]
    max_k, _ = torch.topk(x_flat, k, dim=2)
    
    # Get bottom-k minimum values: [B, D, k]  
    min_k, _ = torch.topk(x_flat, k, dim=2, largest=False)
    
    # Concatenate max and min: [B, D, 2*k]
    pooled = torch.cat([max_k, min_k], dim=2)
    
    # Average across the k values: [B, D]
    return pooled.mean(dim=2)


def spatial_pooling(x: torch.Tensor, method: str = "min_max_k", k: int = 20) -> torch.Tensor:
    """
    Unified spatial pooling with multiple method options.
    
    Args:
        x (torch.Tensor): Input tensor, shape [B, D, H, W]
        method (str): Pooling method. Currently only "min_max_k" is supported.
        k (int): k parameter for min_max_k method. Defaults to 20.
    
    Returns:
        torch.Tensor: Pooled features, shape [B, D]
    """
    if method == "min_max_k":
        return min_max_k_pooling(x, k=k)
    else:
        raise ValueError(f"Unknown pooling method: {method}. Options: min_max_k")


# ==================== Loss Functions ====================

def clip_style_contrastive_loss(z_img: torch.Tensor, z_txt: torch.Tensor, temperature: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes symmetric CLIP-style contrastive loss for image-text alignment.

    Args:
        z_img (torch.Tensor): Image embeddings, shape `[B, D]`.
        z_txt (torch.Tensor): Text embeddings, shape `[B, D]`.
        temperature (Optional[torch.Tensor]): Temperature scaling factor. If None, uses 1/0.07.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of (image_to_text_loss, text_to_image_loss).
    """
    if temperature is None:
        temperature = torch.tensor(1.0 / 0.07, device=z_img.device)

    # Compute similarity matrices in both directions
    logits_ii = (z_img @ z_txt.t()) * temperature  # [B, B]
    logits_tt = logits_ii.t()  # [B, B]

    # Create targets: each embedding should match itself (diagonal)
    target = torch.arange(z_img.size(0), device=z_img.device)

    # Compute cross-entropy losses for both directions
    loss_i2t = F.cross_entropy(logits_ii, target)
    loss_t2i = F.cross_entropy(logits_tt, target)

    return loss_i2t, loss_t2i


# ==================== Training Orchestrator ====================

class FlowTrainer:
    """
    Trainer for text-conditioned segmentation with optional ODE flow alignment.

    This trainer orchestrates:
    - CLIP-style contrastive loss (w_nce): Global image-text alignment
    - Rectified Flow loss (w_rf): Time-conditioned text→visual alignment (if use_text_flow=True)
    """
    def __init__(self, backbone: DinoV3HFBackbone, head: TextCondHead, txt: CLIPTextEncoder,
                 lr: float = 1e-4, wd: float = 0.01, T_max: int = 50,
                 w_nce: float = 1.0, w_rf: float = 1.0, w_text_consistency: float = 0.0, grad_clip: float = 1.0):
        """
        Initializes the FlowTrainer.

        Args:
            backbone (DinoV3HFBackbone): Frozen DINOv3 backbone for feature extraction.
            head (TextCondHead): Text-conditioned segmentation head.
            txt (CLIPTextEncoder): Frozen CLIP text encoder for text feature extraction.
            lr (float): Learning rate for AdamW optimizer. Defaults to 1e-4.
            wd (float): Weight decay for regularization. Defaults to 0.01.
            T_max (int): Maximum number of iterations for CosineAnnealingLR scheduler. Defaults to 50.
            w_nce (float): Weight for CLIP-style contrastive loss. Defaults to 1.0.
            w_rf (float): Weight for Rectified Flow loss and end-point loss. Defaults to 1.0.
            w_text_consistency (float): Weight for text flow backward consistency loss. Defaults to 0.0.
            grad_clip (float): Gradient clipping threshold. If 0, no clipping is applied. Defaults to 1.0.
        """
        # Store core components (backbone and txt encoder are frozen)
        self.backbone, self.head, self.txt = backbone, head, txt

        # Loss function weights
        self.w_nce = w_nce
        self.w_rf = w_rf
        self.w_text_consistency = w_text_consistency
        self.grad_clip = grad_clip

        # Collect parameters to optimize
        params = list(self.head.text_proj.parameters())

        # cls_flow_init is always trained regardless of use_cls_flow
        params += list(self.head.cls_flow_init.parameters())

        if getattr(self.head, 'use_cls_flow', False):
            params += list(self.head.cls_flow_net.parameters())
        elif getattr(self.head, 'use_cls_mlp', False):
            params += list(self.head.cls_proj_mlp.parameters())
        
        # Setup optimizer with AdamW
        self.opt = torch.optim.AdamW(
            params, lr=lr, weight_decay=wd, betas=(0.9, 0.999), eps=1e-8
        )

        # Cosine annealing learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt, T_max=T_max, eta_min=lr * 0.01
        )

        # Mixed precision training scaler
        self.scaler = GradScaler()

    def _encode_texts(self, names: List[str],
                      cached_clip_embeddings: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encodes text names into L2-normalized visual-space embeddings.

        Args:
            names (List[str]): List of text strings (e.g., class names, noun phrases).
            cached_clip_embeddings (Optional[torch.Tensor]): Pre-computed CLIP embeddings
                [K, 768]. If provided, skips CLIP encoding.

        Returns:
            torch.Tensor: L2-normalized text features in visual space, shape `[K, 1024]`.
        """
        if not names:
            device = self.head.text_proj.weight.device
            return torch.empty(0, 1024, device=device)

        if cached_clip_embeddings is not None:
            T_clip = cached_clip_embeddings
        else:
            with torch.no_grad():
                # Encode each name with multiple prompt variations and average
                T_clip = torch.stack([
                    self.txt.encode(build_prompts(n), aggregate="mean").squeeze(0)
                    for n in names
                ], dim=0)  # [K, 768] - CLIP embeddings

        # Project from CLIP space (768D) to visual space (1024D) and normalize
        return l2norm(self.head.text_proj(T_clip))  # [K, 1024]

    def _encode_texts_captions_batch(self, captions: List[List[str]],
                                     caption_clip_embeddings: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        """
        Batch-encodes captions for all images at once for efficiency.
        Each image's multiple captions are averaged first, so N = batch_size.

        Args:
            captions (List[List[str]]): List of caption lists, one per image in batch.
            caption_clip_embeddings (Optional[torch.Tensor]): Pre-computed CLIP embeddings
                for each image's captions (already averaged), shape [B, 768].
                If provided, skips CLIP encoding entirely.

        Returns:
            List[torch.Tensor]: List of caption features, each [2*D] for valid images,
                               or empty tensor for images without captions.
        """
        device = self.head.text_proj.weight.device
        B = len(captions)
        
        # Encode captions per image and average them first
        T_clip_per_image = []
        valid_indices = []
        for i, caps in enumerate(captions):
            if caps and len(caps) > 0:
                if caption_clip_embeddings is not None and caption_clip_embeddings[i].norm() > 1e-6:
                    # Use pre-computed CLIP embedding (skip CLIP encoder)
                    T_clip_per_image.append(caption_clip_embeddings[i])  # [768]
                else:
                    # Fallback: encode with CLIP
                    with torch.no_grad():
                        caps_clip = self.txt.encode(caps, aggregate="mean")  # [num_caps, 768] -> [768] (mean)
                    T_clip_per_image.append(caps_clip.squeeze(0))  # [768]
                valid_indices.append(i)
            else:
                T_clip_per_image.append(None)
        
        # If no captions at all, return empty tensors
        if not valid_indices:
            return [torch.empty(0, 2048, device=device) for _ in range(B)]
        
        # Stack valid caption embeddings: [B_valid, 768] where B_valid = number of images with captions
        self.T_clip_valid = torch.stack([T_clip_per_image[i] for i in valid_indices], dim=0)  # [B_valid, 768]

        # Batch projection (with or without flow, context-free)
        # Don't use CLS token for flow - use context-free flow
        text_features_valid = l2norm(self.head.text_proj(self.T_clip_valid))  # [B_valid, 1024]
        
        # Create results list, filling in valid features and empty tensors for invalid
        results = []
        valid_idx = 0
        for i in range(B):
            if i in valid_indices:
                # Get the feature for this image
                feature = text_features_valid[valid_idx]  # [1024]
                feature = l2norm(feature.unsqueeze(0)).squeeze(0)  # re-normalize
                # Duplicate for 2*D dimension
                feature_2d = torch.cat([feature, feature.detach()], dim=0)  # [2048]
                results.append(feature_2d)
                valid_idx += 1
            else:
                results.append(torch.empty(0, 2048, device=device))

        return results


    def train_step(self, images: torch.Tensor, name_lists: List[List[str]], grid: int = 16,
                   iteration: int = 0, masks: Optional[torch.Tensor] = None,
                   pseudo_masks: Optional[torch.Tensor] = None,
                   pseudo_mask_indices: Optional[List[int]] = None,
                   captions: Optional[List[List[str]]] = None,
                   caption_clip_embeddings: Optional[torch.Tensor] = None,
                   cached_backbone_features: Optional[tuple] = None) -> Dict[str, torch.Tensor]:
        """
        Performs a single training step with multiple loss functions.

        Args:
            images (torch.Tensor): A batch of input images.
            name_lists (List[List[str]]): A list of lists, where each inner list contains
                                        the text concepts for an image in the batch.
            grid (int): The grid size for region-based feature extraction.
            iteration (int): The current training iteration number.
            masks (Optional[torch.Tensor]): Ground truth segmentation masks.
            pseudo_masks (Optional[torch.Tensor]): Generated pseudo-label masks.
            pseudo_mask_indices (Optional[List[int]]): Indices mapping pseudo-masks to their concepts.
            captions (Optional[List[List[str]]]): Optional captions for each image in the batch.
            caption_clip_embeddings (Optional[torch.Tensor]): Pre-computed CLIP caption embeddings.
            cached_backbone_features (Optional[tuple]): Pre-computed (A, cls_token_raw) from backbone.

        Returns:
            Dict[str, torch.Tensor]: A dictionary of computed loss values for logging.
        """
        self.head.train()
        self.opt.zero_grad()

        if cached_backbone_features is not None:
            A, cls_token_raw = cached_backbone_features
        else:
            with torch.no_grad():
                A, cls_token_raw = self.backbone.forward_grid(images)
        device = A.device
        B, C, H, W = A.shape
        
        # Forward pass through the head's visual processing
        A_processed, Z_map = self.head.process_visual_features(A)
        
        # Project CLS token for NCE loss (with optional ODE flow via apply_cls_flow)
        cls_token = self.head.apply_cls_flow(cls_token_raw)  # [B, 1024]
        
        # Add caption features if provided (batch processing for efficiency)
        # Don't pass cls_token to flow - use context-free flow
        caption_sets = None
        if captions is not None and len(captions) > 0:
            caption_sets = self._encode_texts_captions_batch(captions, caption_clip_embeddings=caption_clip_embeddings)
        
        # ===== Loss Calculations =====
        L_nce = torch.tensor(0.0, device=device)
        L_rf = torch.tensor(0.0, device=device)  # Rectified Flow loss
        L_end = torch.tensor(0.0, device=device)  # End-point (geodesic) alignment loss
        L_text_consistency = torch.tensor(0.0, device=device)  # Text flow backward consistency
        
        if caption_sets is not None and len(caption_sets) > 0:
            # Pool visual features using spatial pooling and concatenate with cls_token
            _z_img_pooled = F.normalize(spatial_pooling(Z_map, method="min_max_k", k=self.head.topk), dim=1)  # [B, D]
            # _z_img_pooled = F.normalize(F.adaptive_avg_pool2d(A, (1, 1)).squeeze(-1).squeeze(-1), dim=1)  # [B, D]
            _cls_token = F.normalize(cls_token, dim=1)  # [B, D]
            z_img_with_cls = torch.cat([_z_img_pooled, _cls_token], dim=1)  # [B, 2*D]
            
            # Stack caption features: caption_sets is a list of [2048] tensors
            # Create a mask for samples with valid captions
            valid_mask = torch.tensor([len(caps) > 0 for caps in caption_sets], device=device)
            
            if valid_mask.any():
                cap_text_pooled = torch.stack(caption_sets, dim=0)  # [num_valid, 2048]
                # import pdb; pdb.set_trace()

                # Compute InfoNCE loss only for samples with captions
                Li, Lt = clip_style_contrastive_loss(
                    z_img_with_cls[valid_mask], 
                    cap_text_pooled, 
                    temperature=1.0 / self.head.tau
                )
                L_nce = 0.5 * (Li + Lt)

                # --- Rectified Flow Loss for text→visual alignment (only if use_text_flow=True) ---
                if self.head.use_text_flow and len(valid_captions := [caps for caps, valid in zip(captions, valid_mask.tolist()) if valid and caps]) > 0:
                # Raw projection: only text_flow_init, NO ODE flow
                    _text_raw = l2norm(self.head.text_flow_init(self.T_clip_valid))  # [N, 1024]

                    # Targets: patch_pooled (apply valid_mask)
                    _z_img_pooled_valid = _z_img_pooled[valid_mask]  # [num_valid, 1024]
                    
                    num_valid = _text_raw.shape[0]
                    
                    # Random time for each sample
                    t = torch.rand(num_valid, 1, device=device)  # [num_valid, 1]
                    
                    # === RF Loss: text → patch_pooled (Slerp on hypersphere) ===
                    # z0, z1: already unit
                    z0 = _text_raw
                    z1 = _z_img_pooled_valid

                    dot = torch.sum(z0 * z1, dim=-1, keepdim=True).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
                    alpha = torch.acos(dot)
                    sin_alpha = torch.sin(alpha) + 1e-10

                    # Slerp: z_t (on sphere)
                    w0 = torch.sin((1 - t) * alpha) / sin_alpha
                    w1 = torch.sin(t * alpha) / sin_alpha
                    z_t = w0 * z0 + w1 * z1

                    # Target velocity: derivative of Slerp (already tangent to sphere)
                    v_tgt = (alpha / sin_alpha) * (
                        -torch.cos((1 - t) * alpha) * z0 + torch.cos(t * alpha) * z1
                    )
                        
                    # Model prediction + tangent projection
                    v_pred = self.head.text_flow_net(z_t, t.squeeze(-1))
                    v_pred = v_pred - (v_pred * z_t).sum(-1, keepdim=True) * z_t

                    L_rf = F.mse_loss(v_pred, v_tgt)

                    # --- Text flow backward consistency: backward from image → z0_recon ≈ z0 (text) ---
                    z0_recon = self.head.apply_text_flow_backward(z1.detach())  # z1 = _z_img_pooled_valid [N, 1024]
                    L_text_consistency = (1.0 - (z0_recon * z0).sum(dim=-1).mean()).clamp(min=0.0)

        
        # Total loss
        loss = self.w_nce * L_nce + self.w_rf * (L_rf + L_end) + self.w_text_consistency * L_text_consistency

        # Optimize
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.opt)
        
        # Apply gradient clipping if grad_clip > 0
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(list(self.head.parameters()), self.grad_clip)
        
        self.scaler.step(self.opt)
        self.scaler.update()

        # Step the scheduler after each optimizer step
        if self.scheduler:
            self.scheduler.step()

        return {
            "loss": loss.detach(),
            "L_nce": (self.w_nce * L_nce).detach(),
            "L_rf": (self.w_rf * L_rf).detach(),
            "L_end": (self.w_rf * L_end).detach(),
            "L_text_consistency": (self.w_text_consistency * L_text_consistency).detach(),
        }

    def state_dict(self) -> Dict[str, Any]:
        """Returns the state dictionary of the trainer, including optimizer and scheduler states."""
        return {
            'opt_state_dict': self.opt.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict(),
            'w_nce': self.w_nce,
            'w_rf': self.w_rf,
            'w_text_consistency': self.w_text_consistency,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Loads the state dictionary into the trainer, restoring optimizer and scheduler states."""
        if 'opt_state_dict' in state_dict:
            self.opt.load_state_dict(state_dict['opt_state_dict'])
        if 'scheduler_state_dict' in state_dict and state_dict['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(state_dict['scheduler_state_dict'])
        if 'scaler_state_dict' in state_dict:
            self.scaler.load_state_dict(state_dict['scaler_state_dict'])
        if 'w_nce' in state_dict:
            self.w_nce = state_dict['w_nce']
        if 'w_rf' in state_dict:
            self.w_rf = state_dict['w_rf']
        if 'w_text_consistency' in state_dict:
            self.w_text_consistency = state_dict['w_text_consistency']

    def apply_pamr(self, image: torch.Tensor, mask: torch.Tensor,
                   pamr_iter: int = 10,
                   pamr_kernel: Optional[List[int]] = None) -> torch.Tensor:
        """Refine a soft mask with PAMR (Pixel Adaptive Mask Refinement).

        Lazily initialises the PAMR module on first call and reuses it afterwards.
        Follows the Talk2DINO pattern (dinotext.py) exactly.

        Args:
            image (torch.Tensor): RGB image [B, 3, H_img, W_img] in any float range.
                                  Will be resized internally to match mask resolution.
            mask  (torch.Tensor): Soft mask [B, C, H, W] (logits or probabilities).
            pamr_iter   (int):   Number of PAMR message-passing iterations. Default: 10.
            pamr_kernel (list):  Dilation sizes. Default: [1, 2, 4, 8, 12, 24].

        Returns:
            torch.Tensor: Refined mask [B, C, H, W] (same spatial size as input mask).
        """
        if pamr_kernel is None:
            pamr_kernel = [1, 2, 4, 8, 12, 24]

        if not hasattr(self, '_pamr') or self._pamr is None:
            from .pamr import PAMR as _PAMRClass
            self._pamr = _PAMRClass(pamr_iter, pamr_kernel)
            self._pamr.eval()
            device = next(self.head.parameters()).device
            self._pamr = self._pamr.to(device)

        # Resize image to mask spatial size (same as apply_pamr in dinotext.py)
        image = F.interpolate(image, size=mask.shape[-2:], mode="bilinear", align_corners=True)
        return self._pamr(image, mask)