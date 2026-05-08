import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision
import torch

from .layer import PerturbedLayer

class ExpandLayer(PerturbedLayer):
    def __init__(self, image: torch.Tensor, max_displacement=0.3):
        image = image.contiguous().clone()
        super().__init__(image)
        
        self.max_displacement = max_displacement

        # Use a Linear layer instead of bmm/expand
        self.image_layer = nn.Linear(self.num_pixels, self.C, bias=False)
        with torch.no_grad():
            self.image_layer.weight.copy_(self.flat_image.squeeze(0))
            self.image_layer.weight.requires_grad_(False)

        xs = torch.arange(self.W, dtype=torch.float32)
        ys = torch.arange(self.H, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

        self.register_buffer("x_coords", xs)
        self.register_buffer("y_coords", ys)
        
        # Base coordinates for the target grid flattened to [1, N]
        self.register_buffer("base_x", grid_x.reshape(1, -1))
        self.register_buffer("base_y", grid_y.reshape(1, -1))
        
        # Generate and register the base displacement pattern
        self.register_buffer("base_displacement", self._create_base_displacement(grid_x, grid_y))

    def _create_base_displacement(self, grid_x, grid_y) -> torch.Tensor:
        """Creates a fixed [1, 2, N] displacement field pattern to be scaled by w."""
        base_disp = torch.zeros((1, 2, self.num_pixels))
        
        center_x, center_y = (self.W - 1) / 2.0, (self.H - 1) / 2.0
        
        # Negative sign flips the sampling so the object expands instead of shrinking
        base_disp[0, 0, :] = (-(grid_x - center_x) * self.max_displacement).reshape(-1)
        base_disp[0, 1, :] = (-(grid_y - center_y) * self.max_displacement).reshape(-1)
            
        return base_disp

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        """
        w: [B, 1] tensor representing the intensity of the deformation.
        """
        # Scale the base displacement field by the scalar w
        displacement_field = w.unsqueeze(1) * self.base_displacement # [B, 2, N]
        
        dx = displacement_field[:, 0, :] # [B, N]
        dy = displacement_field[:, 1, :] # [B, N]

        # Calculate source coordinates for bilinear sampling
        src_x = self.base_x + dx
        src_y = self.base_y + dy

        samples = self._bilinear_sample(src_x, src_y)
        return samples.view(w.shape[0], self.C, self.H, self.W)

if __name__ == "__main__":
    torch.manual_seed(37)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = torchvision.datasets.MNIST(root='data', download=True, transform=transform, train=False)
    dataloader = DataLoader(dataset, batch_size=5, shuffle=True)
    
    for img, _ in dataloader:
        img_tensor = img.squeeze(0)[1]
        break
    
    print("Image tensor shape:", img_tensor.shape, (torch.min(img_tensor).item(), torch.max(img_tensor).item()))
        
    # Define an interval of scalar inputs w
    w_values = torch.tensor([[0.0], [0.2], [0.4], [0.6], [0.8], [1.0]]) # [B, 1]
        

    layer = ExpandLayer(
        image=img_tensor, 
        max_displacement=0.3
    )
    perturbed = layer(w_values)
    print(f"{perturbed.shape=}")
    for i, _ in enumerate(perturbed):
        print(f'\t+ w={w_values[i].item():.02f}, sum={_.sum().item():.02f}, min={torch.min(_).item():.02f}, max={torch.max(_).item():.02f}')


    images = [('Original', img_tensor)]
    for i in range(len(w_values)):
        images.append((f'w = {w_values[i].item():.01f}', perturbed[i]))

    fig, axes = plt.subplots(1, len(images), figsize=(12, 3))
        
    for i, (title, img) in enumerate(images):
        axes[i].imshow(img.permute(1, 2, 0).cpu().numpy())
        axes[i].set_title(title)
        axes[i].axis('off')

    plt.suptitle("Expansion", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('figures/expand_layer.png', dpi=300, bbox_inches='tight')
