import torch
import torch.nn.functional as F


def greedy_search(logits: torch.Tensor) -> torch.Tensor:
    """
    Select the token with the largest logit
    """
    return torch.argmax(logits, dim=-1)


def sample_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    Takes logits and converts them to probabilities and samples from thier distribution
    """
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def top_k_sampling(logits: torch.Tensor, k: int) -> torch.Tensor:
    """
    Returns new logits with all values, except for the k largest, set to -inf
    """
    assert k >= 1, f"k was set to {k}, k must be positive"
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    indices_to_remove = sorted_indices[..., k:]

    # Set the probability of removed indices to 0 (by setting logits to -inf)
    cloned_logits = logits.clone()
    neg_infinity = torch.ones_like(logits) * float("-inf")
    cloned_logits.scatter_(dim=-1, index=indices_to_remove, src=neg_infinity)

    return cloned_logits


def top_p_sampling(logits: torch.Tensor, p: float):
    """
    Perform top-p (nucleus) sampling on logits.

    Args:
    logits: torch.Tensor of shape (..., vocab_size)
    p: float, cumulative probability threshold

    Returns:
    torch.Tensor of the same shape as logits, with values outside the top-p set to -inf
    """
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > p

    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    # Scatter sorted tensors to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
    )

    cloned_logits = logits.clone()
    cloned_logits[indices_to_remove] = float("-inf")

    return cloned_logits


def temperature_sampling(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Scales logits by temprature
    """
    return logits / temperature
