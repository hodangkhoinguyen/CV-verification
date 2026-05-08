import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision
import torch

from .layer import PerturbedLayer

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

if __name__ == "__main__":
    torch.manual_seed(37)
    theta_degrees = torch.tensor([0.0, 30.0, 45.0, 60.0, 90.0]).view(-1, 1)
    
    # dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = torchvision.datasets.MNIST(root='data', download=True, transform=transform, train=False)
    dataloader = DataLoader(dataset, batch_size=5, shuffle=True)
    
    for img, _ in dataloader:
        img_tensor = img.squeeze(0)[1]
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

