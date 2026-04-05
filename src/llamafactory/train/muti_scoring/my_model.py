import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from transformers.modeling_outputs import SequenceClassifierOutput, CausalLMOutputWithPast
from transformers import PreTrainedModel
from ...model import load_model
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union
from ...extras.logging import get_logger
import numpy as np

logger = get_logger(__name__)


class CopulaGaussianCorrelation(nn.Module):

    
    def __init__(self, num_dimensions: int = 5):
        super().__init__()
        self.num_dimensions = num_dimensions
        
        # Learnable lower triangular matrix L (num_dimensions x num_dimensions)
        # We only store the lower triangular part to ensure valid Cholesky decomposition
        self.L_lower = nn.Parameter(torch.zeros(num_dimensions, num_dimensions))
        
        # Pooling layer to aggregate features before correlation computation
        self.pooling = nn.AdaptiveAvgPool1d(1)
        
        # Linear transformation layer
        self.linear_transform = nn.Linear(num_dimensions, num_dimensions)
        
        # Initialize L as identity matrix (no correlation initially)
        self._initialize_L()
    
    def _initialize_L(self):
        """Initialize L as a lower triangular matrix close to identity"""
        with torch.no_grad():
            # Initialize diagonal elements to 1.0 (positive for valid Cholesky)
            torch.nn.init.constant_(self.L_lower, 0.0)
            for i in range(self.num_dimensions):
                self.L_lower.data[i, i] = 1.0
    
    def get_lower_triangular(self):
        """
        Extract lower triangular matrix from L_lower parameter.
        Ensures positive diagonal elements for valid Cholesky decomposition.
        """
        # Create lower triangular matrix
        L = torch.tril(self.L_lower)
        
        # Ensure positive diagonal elements (required for Cholesky)
        diag = torch.diag(L)
        diag_clamped = F.softplus(diag) + 1e-6  # softplus ensures positivity
        L = L - torch.diag(diag) + torch.diag(diag_clamped)
        
        return L
    
    def compute_correlation_matrix(self):
        """
        Compute correlation matrix R = L · L^T
        This ensures R is positive semi-definite and valid correlation matrix.
        """
        L = self.get_lower_triangular()
        R = L @ L.t()
        
        # Normalize to correlation matrix (diagonal = 1)
        diag = torch.sqrt(torch.diag(R))
        R_normalized = R / (diag.unsqueeze(0) * diag.unsqueeze(1) + 1e-8)
        
        return R_normalized
    
    def compute_copula_gaussian_gt(self, reg_labels: torch.Tensor) -> torch.Tensor:
        """
        Compute Ground Truth correlation matrix using Copula Gaussian modeling.
        
        This uses rank-based correlation (Spearman-like) which is robust to non-linear dependencies.
        
        Args:
            reg_labels: Ground truth labels [B, 5]
            
        Returns:
            GT correlation matrix [5, 5]
        """
        batch_size = reg_labels.shape[0]
        
        # Convert to ranks (Copula transformation)
        # This maps marginal distributions to uniform [0, 1]
        ranks = torch.zeros_like(reg_labels)
        for i in range(batch_size):
            for dim in range(self.num_dimensions):
                # Compute rank for each dimension
                sorted_vals, sorted_idx = torch.sort(reg_labels[i, :])
                rank = torch.argsort(sorted_idx).float() / (self.num_dimensions - 1)
                ranks[i, dim] = rank[dim]
        
        # Transform to Gaussian space using inverse normal CDF (probit)
        # Add small epsilon to avoid inf
        epsilon = 1e-6
        gaussian_values = torch.erfinv(2 * ranks.clamp(epsilon, 1 - epsilon) - 1) * np.sqrt(2)
        
        # Compute empirical correlation matrix
        # Center the data
        gaussian_values = gaussian_values - gaussian_values.mean(dim=0, keepdim=True)
        
        # Compute correlation: R = X^T X / (N-1)
        gt_corr = (gaussian_values.t() @ gaussian_values) / (batch_size - 1)
        
        # Normalize to ensure diagonal = 1
        diag = torch.sqrt(torch.diag(gt_corr))
        gt_corr_normalized = gt_corr / (diag.unsqueeze(0) * diag.unsqueeze(1) + 1e-8)
        
        return gt_corr_normalized
    
    def forward(self, pooled_features: torch.Tensor, reg_labels: Optional[torch.Tensor] = None):
        """
        Forward pass through Copula Gaussian Correlation module.
        
        Args:
            pooled_features: Pooled features from backbone [B, hidden_size]
            reg_labels: Ground truth labels for computing GT correlation [B, 5] (optional during inference)
            
        Returns:
            reg_logits: Enhanced predictions [B, 5]
            corr_loss: Correlation constraint loss (scalar)
            pred_corr: Predicted correlation matrix [5, 5]
        """
        features_transformed = self.linear_transform(pooled_features)  # [B, 5]
        
        pred_corr = self.compute_correlation_matrix()  # [5, 5]
        
        reg_logits = features_transformed  # Base predictions
        
        # Compute correlation loss if GT labels are provided
        corr_loss = None
        if reg_labels is not None:
            # Compute GT correlation matrix using Copula Gaussian
            gt_corr = self.compute_copula_gaussian_gt(reg_labels)  # [5, 5]
            
            # Constraint loss: minimize difference between predicted and GT correlation
            # Using Frobenius norm
            corr_loss = F.mse_loss(pred_corr, gt_corr)
        
        return reg_logits, corr_loss, pred_corr


import torch.nn.functional as F

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from ..data.template import Template
    from ..hparams import FinetuningArguments, ModelArguments


class my_qwen(PreTrainedModel):

    _supports_sdpa = True
    _supports_flash_attn_2 = True
    supports_gradient_checkpointing = True
    def __init__(
            self,
            tokenizer=None,
            model_args=None,
            finetuning_args=None,
            training_args=None,
    ):
        self.reg_weight = 0.05  # TODO: make this a hyperparameter later

        # Load model first to get config
        wrapper = load_model(
            tokenizer, model_args, finetuning_args,
            is_trainable=training_args.do_train,
        #    full_determinism=training_args.full_determinism
        )
        
        # Save wrapper immediately for later use
        self.wrapper_model = wrapper
        
        # Call parent class initialization
        super().__init__(wrapper.config)
        
        self.sequence_parallel_group = None
        
        # Verify model structure
        assert hasattr(self.wrapper_model, "model"), f"expect ForConditionalGeneration, got {type(self.wrapper_model)}"
        
        # Verify visual module exists
        if hasattr(self.backbone, "model"):
            # Qwen3VLForConditionalGeneration -> .model -> .visual
            assert hasattr(self.backbone.model, "visual"), f"backbone.model has no visual, type={type(self.backbone.model)}"
        elif hasattr(self.backbone, "visual"):
            # Has visual attribute directly
            pass
        else:
            logger.warning(f"Cannot find visual module in backbone type={type(self.backbone)}")

        # Print model structure info (for debugging)
        print("wrapper (wrapper_model) type:", type(self.wrapper_model))
        print("backbone type:", type(self.backbone))

        # Get hidden_size (should be 4096)
        hidden_size = self.backbone.config.hidden_size
        
        # Copula Gaussian Correlation module for modeling 5D dependencies
        self.copula_gaussian = CopulaGaussianCorrelation(num_dimensions=5)

        # Add your custom fully connected layer (e.g., 4096 -> 5)
        self.extra_fc = nn.Sequential(
            # nn.Dropout(p=0.2),
            nn.Linear(hidden_size, 5)  # num_labels = 5
        )

        # Initialize additional layers
        # self.post_init()  # HF recommended initialization method

    @property
    def backbone(self):
        return self.wrapper_model.model

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
        if hasattr(self.wrapper_model, "gradient_checkpointing_enable"):
            self.wrapper_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
        self.config.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        if hasattr(self.backbone, "gradient_checkpointing_disable"):
            self.backbone.gradient_checkpointing_disable()
        if hasattr(self.wrapper_model, "gradient_checkpointing_disable"):
            self.wrapper_model.gradient_checkpointing_disable()
        self.config.gradient_checkpointing = False

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            pixel_values: Optional[torch.Tensor] = None,  # Required for multimodal!
            attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,  # SFT token labels
            reg_labels=None,  # New: regression labels [B,5] (all zeros for debugging)
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            copula_weight: float = 0.1,  # Weight for correlation constraint loss
            **kwargs,
    ) -> Dict[str, torch.Tensor]:

        # Call underlying model, force return hidden_states
        sft_out = self.wrapper_model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,  # Must be True to extract features
            return_dict=True,
            **kwargs,
        )

        sft_loss = sft_out.loss
        sft_logits = sft_out.logits

        # Safely get hidden_states
        if not hasattr(sft_out, 'hidden_states') or sft_out.hidden_states is None:
            raise RuntimeError("Model output does not contain hidden_states. Please check output_hidden_states=True.")
        
        h = sft_out.hidden_states[-1]  # [B,L,H]

        # Extract features from the last valid token
        if attention_mask is not None:
            last_idx = (attention_mask.sum(dim=1) - 1).clamp(min=0)
            bidx = torch.arange(h.size(0), device=h.device)
            pooled = h[bidx, last_idx]
        else:
            pooled = h[:, -1, :]

        # Pass through Copula Gaussian module to get enhanced predictions and correlation loss
        reg_logits, corr_loss, pred_corr = self.copula_gaussian(pooled, reg_labels)
        
        # Also get base predictions from FC layer (for comparison or alternative use)
        # base_reg_logits = self.extra_fc(pooled)  # [B,5]

        # Compute regression loss (using Copula-enhanced logits)
        reg_loss = None
        if reg_labels is not None:
            reg_labels = reg_labels.to(reg_logits.device).to(reg_logits.dtype)
            reg_loss = F.binary_cross_entropy_with_logits(reg_logits, reg_labels)

        # Combine all losses
        loss = (1 - self.reg_weight) * sft_loss + self.reg_weight * reg_loss
        
        # Add correlation constraint loss if available
        if corr_loss is not None:
            loss = loss + copula_weight * corr_loss
        
        # Build output dictionary
        out = {
            "loss": loss,
            "logits": sft_logits,
            "reg_logits": reg_logits,
            "pred_corr": pred_corr,  # Predicted correlation matrix [5, 5]
        }

        if sft_loss is not None:
            out["sft_loss"] = sft_loss.detach()
        if reg_loss is not None:
            out["reg_loss"] = reg_loss.detach()
        if corr_loss is not None:
            out["corr_loss"] = corr_loss.detach()
        
        return out