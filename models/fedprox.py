import torch
import torch.nn as nn
from collections import OrderedDict
from typing import List, Dict

def compute_proximal_loss(model: nn.Module, global_params: List[torch.Tensor], mu: float = 0.001) -> torch.Tensor:
    """
    Computes the FedProx proximal penalty term: (mu / 2) * ||local_weights - global_weights||^2
    """
    proximal_term = torch.tensor(0.0, device=next(model.parameters()).device)
    for local_p, global_p in zip(model.parameters(), global_params):
        proximal_term = proximal_term + (local_p - global_p.to(local_p.device)).norm(2) ** 2
    return (mu / 2.0) * proximal_term
