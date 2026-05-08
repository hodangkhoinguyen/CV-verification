from torch.utils.data import DataLoader
from typing import Any, Dict
import torch.nn as nn
import numpy as np
import argparse
import random
import torch
import os


            
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_vnnlib_str(data_lb: torch.Tensor, data_ub: torch.Tensor, prediction: torch.Tensor):
    # input bounds
    x_lb = data_lb.flatten()
    x_ub = data_ub.flatten()
    
    # outputs
    n_class = prediction.numel()
    y = prediction.argmax(-1).item()
    
    base_str = f"; Specification for class {int(y)}\n"
    base_str += f"\n; Definition of input variables\n"
    for i in range(len(x_ub)):
        base_str += f"(declare-const X_{i} Real)\n"

    base_str += f"\n; Definition of output variables\n"
    for i in range(n_class):
        base_str += f"(declare-const Y_{i} Real)\n"

    base_str += f"\n; Definition of input constraints\n"
    for i in range(len(x_ub)):
        base_str += f"(assert (<= X_{i} {x_ub[i]:.8f}))\n"
        base_str += f"(assert (>= X_{i} {x_lb[i]:.8f}))\n\n"

    base_str += f"\n; Definition of output constraints\n"
    spec_i = base_str
    spec_i += f"(assert (or\n"
    for i in range(n_class):
        if i == y:
            continue
        spec_i += f"\t(and (>= Y_{i} Y_{y}))\n"
    spec_i += f"))\n"
    return [spec_i]
        


def get_valid_data(args, model, test_loader, label_to_index, device):
    valid_data = []
    sample_per_class = {v: args.sample_per_class for v in label_to_index.values()}
    model.to(device)

    # reduce 4 properties
    for i in range(4):
        sample_per_class[i] = 0

    for x, y in test_loader:
        x = x.to(device)
        y = y.to(device).item()
        if not sample_per_class[y]:
            continue
        logit = model(x)
        pred = logit.argmax(-1).item()
        if pred != y:
            continue
        assert y in sample_per_class
        sample_per_class[y] -= 1
        valid_data.append((x.cpu(), y, logit))
        if sum(sample_per_class.values()) == 0:
            break
    print(f"Found {len(valid_data)=} {[v[1] for v in valid_data]}")
    return valid_data
