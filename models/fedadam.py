import torch
from collections import OrderedDict
from typing import List, Dict

class FedAdamServerOptimizer:
    """
    Server-side adaptive optimization (Reddi et al., 2021).
    Applies Adam moments over aggregated pseudo-gradients.
    """
    def __init__(self, model_params: OrderedDict, server_lr: float = 1e-3, beta1: float = 0.9, beta2: float = 0.99, tau: float = 1e-3):
        self.server_lr = server_lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.tau = tau
        self.m = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in model_params.items()}
        self.v = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in model_params.items()}

    def step(self, global_weights: OrderedDict, aggregated_weights: OrderedDict) -> OrderedDict:
        new_weights = OrderedDict()
        for k in global_weights.keys():
            delta = aggregated_weights[k].float() - global_weights[k].float()
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * delta
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * (delta ** 2)
            update = self.server_lr * self.m[k] / (torch.sqrt(self.v[k]) + self.tau)
            new_weights[k] = global_weights[k] + update
        return new_weights
