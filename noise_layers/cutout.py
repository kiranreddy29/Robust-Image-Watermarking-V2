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
            y_starts = torch.randint(0, h - self.block_size + 1, (b, num_blocks), device=image.device)
            x_starts = torch.randint(0, w - self.block_size + 1, (b, num_blocks), device=image.device)
            b_idx = torch.arange(b, device=image.device).unsqueeze(1).expand(-1, num_blocks)

            for i in range(self.block_size):
                for j in range(self.block_size):
                    mask[b_idx, 0, y_starts + i, x_starts + j] = 0.0

        noised_image = image * mask
        return [noised_image, noised_and_cover[1]]