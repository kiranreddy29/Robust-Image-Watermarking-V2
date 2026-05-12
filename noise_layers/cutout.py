import torch
import torch.nn as nn

class CutoutAttack(nn.Module):
    def __init__(self, drop_prob=0.15, block_size=48):
        super(CutoutAttack, self).__init__()
        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, noised_and_cover):
        image = noised_and_cover[0].clone()
        b, c, h, w = image.shape
        mask = torch.ones((b, 1, h, w), device=image.device)
        num_blocks = int((h * w * self.drop_prob) / (self.block_size ** 2))

        if num_blocks > 0:
            for k in range(num_blocks):
                y = torch.randint(0, h - self.block_size + 1, (b,), device=image.device)
                x = torch.randint(0, w - self.block_size + 1, (b,), device=image.device)
                for bi in range(b):
                    # Optimized tensor slicing instead of double for-loops
                    mask[bi, 0, y[bi]:y[bi]+self.block_size, x[bi]:x[bi]+self.block_size] = 0.0

        return [image * mask, noised_and_cover[1]]