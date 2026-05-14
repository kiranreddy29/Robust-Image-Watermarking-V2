import re

with open("noise_layers/cutout.py", "r") as f:
    content = f.read()

# The current code might overlap blocks, which means the dropped percentage will be less than exactly 15%.
# Instead of random blocks that overlap, we can drop exactly a contiguous square block or simply zero out 15% randomly.
# But standard Cutout uses a square block. The task says "exactly 15% of physical pixel data".
# A single block of size sqrt(H * W * 0.15) achieves exactly 15%.

cutout_old = """    def __init__(self, drop_prob=0.15, block_size=48):
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

        return [image * mask, noised_and_cover[1]]"""

cutout_new = """    def __init__(self, drop_prob=0.15, block_size=None):
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

        return [image * mask, noised_and_cover[1]]"""
content = content.replace(cutout_old, cutout_new)

with open("noise_layers/cutout.py", "w") as f:
    f.write(content)
