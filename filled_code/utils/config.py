from dataclasses import dataclass

@dataclass
class Config:
    d_model: int = 768
    layer_norm_eps: float = 1e-5
    d_vocab: int = 50257
    n_ctx: int = 1024
    d_head: int = 64
    d_mlp: int = 3072
    n_heads: int = 12
    n_layers: int = 12

if __name__ == "__main__":
  cfg = Config()
  print(cfg)