import torch
import copy
from collections import OrderedDict
from typing import List, Dict

def federated_average(client_state_dicts: List[Dict[str, torch.Tensor]], client_sizes: List[int]) -> OrderedDict:
    """
    Computes sample-weighted Federated Averaging (McMahan et al., 2017).
    """
    total_size = sum(client_sizes)
    avg_weights = OrderedDict()
    for key in client_state_dicts[0].keys():
        weighted_sum = sum(
            client_state_dicts[i][key].float() * (client_sizes[i] / total_size)
            for i in range(len(client_state_dicts))
        )
        avg_weights[key] = weighted_sum
    return avg_weights
