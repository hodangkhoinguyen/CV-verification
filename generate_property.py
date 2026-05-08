from torch.utils.data import DataLoader
import warnings
import argparse
import torch
import os

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*TorchScript-based ONNX export.*")

from utils import set_seed, get_device
from property.expand_prop import generate_expand_prop
from property.rotate_prop import generate_rotate_prop
from helper.mnist import get_mnist_datasets
from helper.fashion_mnist import get_fashion_mnist_datasets
from helper.model import FNN2, FNN4

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model", type=str, required=True, choices=["fnn2", "fnn4"])
    p.add_argument("--dataset", type=str, required=True, choices=["mnist", "fashionmnist"])
    p.add_argument("--sample_per_class", type=int, default=1)
    p.add_argument("--timeout", type=int, default=30)
    p.add_argument("--checkpoint_dir", type=str, default="./checkpoints/")
    p.add_argument("--data_dir", type=str, default="./data/", help="Root directory for data")
    p.add_argument("--benchmark_dir", type=str, required=True, help="Root directory for benchmark")
    args = p.parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.benchmark_dir, exist_ok=True)
    return args

@torch.no_grad()
def main():
    args = parse_args()
    set_seed(args.seed)
    
    device = get_device()
    print(f"Using {device=}")
    
    if args.dataset == "mnist":
        dataset_func = get_mnist_datasets
    elif args.dataset == "fashionmnist":
        dataset_func = get_fashion_mnist_datasets
    else:
        raise ValueError(f"{args.data} not found")

    _, val_ds, label_mapping = dataset_func(root=args.data_dir, download=True)
    num_classes = len(label_mapping)

    test_loader = DataLoader(
        val_ds, 
        batch_size=1, 
        shuffle=False,
        num_workers=os.cpu_count(), 
        pin_memory=True, 
        drop_last=False,
    )
    print(f'Dataloaders: {len(val_ds)=}')
    
    if args.model == "fnn2":
        model = FNN2(
            input_size=784,
            hidden_size=128,
            output_size=num_classes,
        )
    elif args.model == "fnn4":
        model = FNN4(
            input_size=784,
            hidden_size=128,
            output_size=num_classes,
        )
    else:
        raise ValueError(f"{args.model} not found")
    print(model)
    model.to(device)
    
    # load checkpoint
    checkpoint_path = os.path.join(args.checkpoint_dir, f"{args.dataset}_{args.model}.pt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    
    generate_rotate_prop(args, model, test_loader, checkpoint["label_to_index"], device)
    generate_expand_prop(args, model, test_loader, checkpoint["label_to_index"], device)

if __name__ == "__main__":
    main()