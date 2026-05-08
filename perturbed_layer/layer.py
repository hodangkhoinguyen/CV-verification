import torch
import torch.nn as nn

class PerturbedLayer(nn.Module):
    def __init__(self, image: torch.Tensor):
        super().__init__()
        self.C, self.H, self.W = image.shape
        self.num_pixels = self.H * self.W
        self.register_buffer("image", image)
        self.register_buffer("flat_image", image.view(1, self.C, -1))

    @staticmethod
    def _tent_weights(offset: torch.Tensor) -> torch.Tensor:
        def abs2relu(x: torch.Tensor) -> torch.Tensor:
            return 2 * torch.relu(x) - x

        z = 1.0 - abs2relu(offset)
        return 0.5 * (z + abs2relu(z))

    def _bilinear_sample(self, src_x: torch.Tensor, src_y: torch.Tensor) -> torch.Tensor:
        num_coords = src_x.shape[1]
        
        # Compute separable triangular weights
        dx = src_x.unsqueeze(-1) - self.x_coords  # [B, N, W]
        dy = src_y.unsqueeze(-1) - self.y_coords  # [B, N, H]

        weights_x = self._tent_weights(dx)  # [B, N, W]
        weights_y = self._tent_weights(dy)  # [B, N, H]

        weights_x = weights_x.view(-1, num_coords, 1, self.W)
        weights_y = weights_y.view(-1, num_coords, self.H, 1)

        weights_xy = weights_y * weights_x  # [B, N, H, W]
        weights_flat = weights_xy.view(-1, num_coords, self.num_pixels)  # [B, N, H*W]

        sampled = self.image_layer(weights_flat)  # [B, N, C]
        sampled = sampled.permute(0, 2, 1)  # [B, C, N]
        return sampled
