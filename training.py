from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import warnings
import argparse
import os
import time
import torch
from collections import defaultdict

warnings.filterwarnings("ignore")

from utils import set_seed, get_device
from helper.model import FNN2, FNN4
from helper.mnist import get_mnist_datasets
from helper.fashion_mnist import get_fashion_mnist_datasets


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True, choices=["fnn2", "fnn4"])
    p.add_argument("--dataset", type=str, required=True, choices=["mnist", "fashionmnist"])
    p.add_argument("--data_dir", type=str, default="./data", help="Root directory for data")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--num_workers", type=int, default=os.cpu_count())
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint_dir", type=str, default="./checkpoints/")
    args = p.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)
    return args

def _step(model, criterion, optimizer, scaler, batch, train: bool = True):
    device = get_device()
    x, y = batch
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)

    if train:
        optimizer.zero_grad(set_to_none=True)

    with torch.amp.autocast(device_type=device.type, enabled=scaler.is_enabled()):
        logits = model(x)              # [B, num_classes]
        loss = criterion(logits, y)    # CrossEntropy

    if train:
        scaler.scale(loss).backward()
        
        # Apply gradient clipping if specified
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        
        scaler.step(optimizer)
        scaler.update()

    with torch.no_grad():
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean().item()

    return {"loss": float(loss.detach().cpu().item()), "acc": acc}

def run_epoch(model, criterion, optimizer, scaler, loader: DataLoader, train: bool):
    model.train(train)
    agg = defaultdict(float)
    n = 0
    for batch in loader:
        metrics = _step(model, criterion, optimizer, scaler, batch, train=train)
        bs = batch[0].size(0)
        for k, v in metrics.items():
            agg[k] += v * bs
        n += bs
    for k in agg:
        agg[k] /= max(n, 1)
    return dict(agg)

def train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, epochs: int, log_interval: int = 1):
    best_val = float("inf")
    best_state = None

    # Optimizer and criterion
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scaler = torch.amp.GradScaler(device=get_device(), enabled=True)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_metrics = run_epoch(model, criterion, optimizer, scaler, train_loader, train=True)
        val_metrics = run_epoch(model, criterion, optimizer, scaler, val_loader, train=False)
        dt = time.time() - t0

        if (epoch % log_interval) == 0:
            print(f"Epoch {epoch:03d} / {epochs:03d} |  "
                    f"train_loss={train_metrics['loss']:.4f} acc={train_metrics['acc']:.4f} | "
                    f"val_loss={val_metrics['loss']:.4f} acc={val_metrics['acc']:.4f} | "
                    f"time={dt:.1f}s")

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_val

def main():
    args = parse_args()
    set_seed(args.seed)
    
    checkpoint_path = os.path.join(args.checkpoint_dir, f"{args.dataset}_{args.model}.pt")
    if os.path.exists(checkpoint_path):
        print(f"Checkpoint found at {checkpoint_path}. Done.")
        return
    
    device = get_device()
    print(f"Using {device=}")

    if args.dataset == "mnist":
        dataset_func = get_mnist_datasets
    elif args.dataset == "fashionmnist":
        dataset_func = get_fashion_mnist_datasets
    else:
        raise ValueError(f"{args.dataset} not found")

    train_ds, val_ds, label_mapping = dataset_func(root=args.data_dir, download=True)
    num_classes = len(label_mapping)
    
    # Dataloaders
    train_loader = DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers, 
        pin_memory=True, 
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers, 
        pin_memory=True, 
        drop_last=False,
    )

    print(f'Dataloaders: {len(train_ds)=} {len(val_ds)=}')
    
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


    print(f"Starting training...")
    train(model, train_loader, val_loader, epochs=args.epochs, log_interval=1)


    # Save checkpoint (same for both tasks)
    checkpoint_data = {
        "model_state": model.state_dict(),
        "args": vars(args),
        "label_to_index": label_mapping,
    }
    
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(checkpoint_data, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}\n")

if __name__ == "__main__":
    main()
