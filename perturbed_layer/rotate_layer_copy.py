import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision
import torch
import argparse
import os
import onnx2pytorch
import onnx

from .layer import PerturbedLayer
from utils import get_valid_data, get_device
from helper.model import FNN2
from helper.mnist import get_mnist_datasets

class RotationLayer(PerturbedLayer):

    def __init__(self, image: torch.Tensor):
        image = image.contiguous().clone()
        super().__init__(image)

        # Use a Linear layer instead of bmm/expand
        self.image_layer = nn.Linear(self.num_pixels, self.C, bias=False)
        with torch.no_grad():
            self.image_layer.weight.copy_(self.flat_image.squeeze(0))
            self.image_layer.weight.requires_grad_(False)

        c_x = (self.W - 1) / 2.0
        c_y = (self.H - 1) / 2.0
        self.register_buffer("center", torch.tensor([c_x, c_y], dtype=torch.float32))

        xs = torch.arange(self.W, dtype=torch.float32)
        ys = torch.arange(self.H, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

        x_rel = (grid_x - c_x).reshape(1, -1)
        y_rel = (grid_y - c_y).reshape(1, -1)
        self.register_buffer("x_rel", x_rel)
        self.register_buffer("y_rel", y_rel)

        self.register_buffer("x_coords", xs)
        self.register_buffer("y_coords", ys)

    def forward(self, theta) -> torch.Tensor:
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        # x' = x*cos + y*sin
        src_x = (cos_theta * self.x_rel) + (sin_theta * self.y_rel) + self.center[0]
        # y' = -x*sin + y*cos
        src_y = (-sin_theta * self.x_rel) + (cos_theta * self.y_rel) + self.center[1]

        samples = self._bilinear_sample(src_x, src_y)
        return samples.view(theta.shape[0], self.C, self.H, self.W)

def main():
    torch.manual_seed(37)
    theta_degrees = torch.tensor([0.0, 30.0, 45.0, 60.0, 90.0]).view(-1, 1)
    
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
    
    layer = RotationLayer(img_tensor)
    theta_radians = torch.deg2rad(theta_degrees)
    perturbed = layer(theta_radians)
    
    print("Output Shape:", perturbed.shape, [_.sum().item() for _ in perturbed])
    
    images = [('Original', img_tensor)]
    for i in range(len(theta_degrees)):
        images.append((f'theta = {theta_degrees[i].item()}', perturbed[i]))

    fig, axes = plt.subplots(1, len(images), figsize=(12, 3))
    for i, (title, img) in enumerate(images):
        axes[i].imshow(img.permute(1, 2, 0).cpu().numpy())
        axes[i].set_title(title)
        axes[i].axis('off')

    plt.suptitle("Rotation", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('figures/rotate_layer.png', dpi=300, bbox_inches='tight')


def visualize_sat():
    print("Visualize SAT results")
    p = argparse.ArgumentParser()
    p.add_argument("--sample_per_class", type=int, default=1)
    args = p.parse_args()
    _, val_ds, label_mapping = get_mnist_datasets(root="/storage/nguyenho/CV-verification/data", download=True)
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
    checkpoint_path = os.path.join("/storage/nguyenho/CV-verification/checkpoints", f"mnist_fnn2.pt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    valid_data = get_valid_data(args, model, test_loader, checkpoint["label_to_index"], get_device())
    # print(len(valid_data))

    idx_list = [5, 9, 11, 12, 13]
    radians = torch.tensor([0.2618, 0.5067, 0.4822, 0.6229, 0.7248]).view(-1, 1)
    degrees = torch.rad2deg(radians)
    print(radians)

    degree_ranges = [
        "0.0-15.0", "15.0-30.0", "30.0-45.0", "45.0-90.0"
    ]

    images = []
    for i, idx in enumerate(idx_list):
        match_idx = idx % 6
        img, y, logit = valid_data[match_idx]
        img_tensor = img.squeeze(0)
        layer = RotationLayer(img_tensor)
        perturbed = layer(radians[i])
        print(perturbed[0].shape)

        onnx_model = onnx.load(f"/storage/nguyenho/CV-verification/benchmark_mnist/rotate_fnn2/onnx/{idx}_42_fnn2_rotate_{degree_ranges[idx // 6]}.onnx")
        pytorch_model = onnx2pytorch.ConvertModel(onnx_model)
        logit = pytorch_model(radians[i])
        pred = logit.argmax(-1).item()

        images.append((f'theta = {degrees[i].item():.2f}, predict = {pred}', perturbed[0]))

    fig, axes = plt.subplots(1, len(images), figsize=(12, 3))
    for i, (title, img) in enumerate(images):
        axes[i].imshow(img.permute(1, 2, 0).cpu().numpy())
        axes[i].set_title(title)
        axes[i].axis('off')

    plt.suptitle("Adversarial examples of fnn2", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('figures/rotate_violation.png', dpi=300, bbox_inches='tight')


    plt.suptitle("Adversarial examples of fnn2", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('figures/rotate_violation_from_onnx.png', dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    # main()
    visualize_sat()

