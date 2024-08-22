import torch as t
from torch import nn
from torch import Tensor

from utils.config import Config

'''
Layer Normalization is a type of normalization that normalizes across the model dimension. Each token in the input sequence is normalized independently of the other tokens in the sequence. Remember that in the transformer, the batches of sequences form a 3D tensor of shape (batch_size, sequence_length, d_model) where each batch contains batch_size sequences of length sequence_length where each token in the sequence is a high-dimensional vector of dim d_model.

(This is in contrast to Batch Normalization, which normalizes the activations of a layer across the batch dimension.)

'''
class LayerNorm(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        # Learnable weight, initialized to 1s
        self.w = nn.Parameter(t.ones(cfg.d_model))
        # Learnable bias, initialized to 0s
        self.b = nn.Parameter(t.zeros(cfg.d_model))

    def forward(self, residual: Float[Tensor, "batch posn d_model"]) -> Float[Tensor, "batch posn d_model"]:
        # Compute mean across the model dimension (the last dimension)
        residual_mean = residual.mean(dim=(-1), keepdim = True) 
        # Compute variance across the model dimension (the last dimension)
        residual_var = residual.var(dim=(-1), keepdim = True, unbiased = False)
        
        # Sidenote: a small epsilon value is added to the variance to prevent division by zero
        residual_std = (residual_var + self.cfg.layer_norm_eps).sqrt()

        # Normalize the residual by taking the difference between the residual and the mean, and dividing by the square root of the variance, then multiply by the learned weight and add the learned bias
        normalized_residual = (residual - residual_mean) / residual_std
        
        # Multiple by learned weights and add learned bias
        return normalized_residual * self.w + self.b  