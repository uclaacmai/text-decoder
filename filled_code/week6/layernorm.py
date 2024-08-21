import torch as t
from torch import nn
from torch import Tensor

from utils.config import Config

'''
Layer Normalization is a type of normalization that normalizes the activations of a layer for each individual sample in a batch across the feature/model dimension. Each token in the input sequence is normalized independently of the other tokens in the sequence. Remember that the input to the transformer is a 3D tensor of shape (batch_size, sequence_length, d_model). The normalization is done across the d_model dimension.

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
        mean = t.mean(residual, dim=(-1), keepdim = True) 
        # Compute variance across the model dimension (the last dimension)
        var = t.var(residual, dim=(-1), keepdim = True, unbiased = False)

        # Normalize the residual by taking the difference between the residual and the mean, and dividing by the square root of the variance, then multiply by the learned weight and add the learned bias
        # An additional note is that a small epsilon value is added to the variance to prevent division by zero
        return ( (residual - mean) / ((var + self.cfg.layer_norm_eps).sqrt()) ) * self.w + self.b
