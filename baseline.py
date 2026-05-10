from torch.utils.data import DataLoader
import os
import torchvision
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt
import onnx
import onnx2pytorch
import argparse
from helper.model import FNN2, FNN4


from utils import get_valid_data, get_device
from helper.mnist import get_mnist_datasets
from helper.fashion_mnist import get_fashion_mnist_datasets

NUM_ATTEMPTS = 250


def visualize_LR():
    torch.manual_seed(37)
    epsilon = [0.1, 0.2, 0.3, 0.5]

    # dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = torchvision.datasets.FashionMNIST(root='data', download=True, transform=transform, train=False)
    dataloader = DataLoader(dataset, batch_size=5, shuffle=True)

    for img, _ in dataloader:
        img_tensor = img.squeeze(0)[3]
        break


    def linf_perturbation(x, eps):
        """Uniform random perturbation bounded by L-inf norm = eps."""
        noise = torch.empty_like(x).uniform_(-eps, eps)
        return torch.clamp(x + noise, -3.0, 3.0)

    images = [('Original', img_tensor)]
    for eps in epsilon:
        perturbed = linf_perturbation(img_tensor, eps)
        images.append((f'ε = {eps}', perturbed))

    fig, axes = plt.subplots(1, len(images), figsize=(14, 3))
    for i, (title, img) in enumerate(images):
        axes[i].imshow(img.permute(1, 2, 0).cpu().numpy())
        axes[i].set_title(title)
        axes[i].axis('off')

    plt.suptitle("L-inf Norm Perturbation", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('figures/linf_perturbation.png', dpi=300, bbox_inches='tight')
    plt.show()

def get_range_radians(degree_range_str):
    lo, hi = map(float, degree_range_str.split("-"))
    return torch.deg2rad(torch.tensor(lo)), torch.deg2rad(torch.tensor(hi))

def run_random_rotation():
    p = argparse.ArgumentParser()
    p.add_argument("--sample_per_class", type=int, default=1)
    args = p.parse_args()
    _, val_ds, label_mapping = get_fashion_mnist_datasets(root="/storage/nguyenho/CV-verification/data", download=True)
    test_loader = DataLoader(
        val_ds, 
        batch_size=1, 
        shuffle=False,
        num_workers=os.cpu_count(), 
        pin_memory=True, 
        drop_last=False,
    )
    num_classes = len(label_mapping)

    model = FNN2(
        input_size=784,
        hidden_size=128,
        output_size=num_classes,
    )
    checkpoint_path = os.path.join("/storage/nguyenho/CV-verification/checkpoints", f"fashionmnist_fnn2.pt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    valid_data = get_valid_data(args, model, test_loader, checkpoint["label_to_index"], get_device())
    # print(len(valid_data))


    degree_ranges = [
        "0.0-15.0", "15.0-30.0", "30.0-45.0", "45.0-90.0"
    ]

    file = open("result_fashion/baseline/fnn2_rotation.csv", "w")
    for idx in range(24):
        match_idx = idx % 6
        img, y, logit = valid_data[match_idx]

        range_str = degree_ranges[idx // 6]
        lo_rad, hi_rad = get_range_radians(range_str)

        random_thetas = torch.empty(NUM_ATTEMPTS, 1).uniform_(lo_rad.item(), hi_rad.item())
        onnx_model = onnx.load(f"/storage/nguyenho/CV-verification/benchmark_fashion/rotate_fnn2/onnx/{idx}_42_fnn2_rotate_{degree_ranges[idx // 6]}.onnx")
        pytorch_model = onnx2pytorch.ConvertModel(onnx_model)
        pytorch_model.eval()

        with torch.no_grad():
            results = []
            for theta in random_thetas:                          # theta: [1]
                logit = pytorch_model(theta)                      # [1, num_classes]
                pred = logit.argmax(dim=1).item()
                results.append(pred == y)
        
        print(f"[idx={idx}] range={range_str} | true_label={y} | "
            f"robust={all(results)} ({sum(results)}/{NUM_ATTEMPTS} correct)")
        file.write(f"{idx}_42_fnn2_rotate_{degree_ranges[idx // 6]}\trobust={all(results)}\t({sum(results)}/{NUM_ATTEMPTS} correct)\n")

    file.close()


def run_random_expand():
    p = argparse.ArgumentParser()
    p.add_argument("--sample_per_class", type=int, default=1)
    args = p.parse_args()
    _, val_ds, label_mapping = get_fashion_mnist_datasets(root="/storage/nguyenho/CV-verification/data", download=True)
    test_loader = DataLoader(
        val_ds, 
        batch_size=1, 
        shuffle=False,
        num_workers=os.cpu_count(), 
        pin_memory=True, 
        drop_last=False,
    )
    num_classes = len(label_mapping)

    model = FNN2(
        input_size=784,
        hidden_size=128,
        output_size=num_classes,
    )
    checkpoint_path = os.path.join("/storage/nguyenho/CV-verification/checkpoints", f"fashionmnist_fnn2.pt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    valid_data = get_valid_data(args, model, test_loader, checkpoint["label_to_index"], get_device())
    # print(len(valid_data))


    expansion_parameter = [
        0.2, 0.5, 0.8
    ]

    file = open("result_fashion/baseline/fnn2_expansion.csv", "w")
    for idx in range(18):
        match_idx = idx % 6
        img, y, logit = valid_data[match_idx]

        hi_expansion = expansion_parameter[idx // 6]
        low_expansion = 0

        random_thetas = torch.empty(NUM_ATTEMPTS, 1).uniform_(low_expansion, hi_expansion)
        onnx_model = onnx.load(f"/storage/nguyenho/CV-verification/benchmark_fashion/expand_fnn2/onnx/{idx}_42_fnn2_{hi_expansion}.onnx")
        pytorch_model = onnx2pytorch.ConvertModel(onnx_model)
        pytorch_model.eval()

        with torch.no_grad():
            results = []
            for theta in random_thetas:                          # theta: [1]
                logit = pytorch_model(theta)                      # [1, num_classes]
                pred = logit.argmax(dim=1).item()
                results.append(pred == y)
        
        file.write(f"{idx}_42_fnn2_{hi_expansion}\trobust={all(results)}\t({sum(results)}/{NUM_ATTEMPTS} correct)\n")

    file.close()


if __name__ == "__main__":
    # visualize_LR()
    run_random_rotation()
    run_random_expand()
