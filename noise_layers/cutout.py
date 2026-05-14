import torch
import torch.nn as nn

class CutoutAttack(nn.Module):
    def __init__(self, drop_prob=0.15, block_size=None):
        super(CutoutAttack, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, noised_and_cover):
        image = noised_and_cover[0].clone()
        b, c, h, w = image.shape
        mask = torch.ones((b, 1, h, w), device=image.device)

        # Calculate exactly 15% of the total pixels as a single square block
        total_pixels = h * w
        drop_pixels = int(total_pixels * self.drop_prob)
        block_h = int(drop_pixels ** 0.5)
        block_w = int(drop_pixels / block_h)

        # In case of rounding errors, adjust block_w slightly to be as close as possible
        # Alternatively, using exact block dimensions is fine.

        y = torch.randint(0, h - block_h + 1, (b,), device=image.device)
        x = torch.randint(0, w - block_w + 1, (b,), device=image.device)
        for bi in range(b):
            mask[bi, 0, y[bi]:y[bi]+block_h, x[bi]:x[bi]+block_w] = 0.0

        return [image * mask, noised_and_cover[1]]