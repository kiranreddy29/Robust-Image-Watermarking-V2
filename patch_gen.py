import re

with open("models/generator.py", "r") as f:
    content = f.read()

# Add LearnedDemodulation class
learned_demodulation_code = """
class LearnedDemodulation(nn.Module):
    def __init__(self, channels=3):
        super().__init__()
        # Input: raw_secret (3 channels) + mask (1 channel) = 4 channels
        self.net = nn.Sequential(
            nn.Conv2d(channels + 1, 32, 3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(32, channels, 3, padding=1)
        )

    def forward(self, secret, mask):
        x = torch.cat([secret, mask], dim=1)
        return self.net(x)

class WatermarkGenerator(nn.Module):
"""

content = content.replace("class WatermarkGenerator(nn.Module):", learned_demodulation_code)

# Add self.demodulator to WatermarkGenerator
init_code = """    def __init__(self, num_blocks=8):
        super().__init__()

        # 1. Initialize your unique saliency feature
        self.saliency     = TextureSaliency()
        self.demodulator  = LearnedDemodulation(channels=3)

        self.isn_blocks   = nn.ModuleList([InvertibleBlock(in_channels=6) for _ in range(num_blocks)])"""

content = re.sub(r'    def __init__\(self, num_blocks=8\):\n        super\(\)\.__init__\(\)\n        \n        # 1. Initialize your unique saliency feature\n        self.saliency     = TextureSaliency\(\) \n        \n        self.isn_blocks   = nn.ModuleList\(\[InvertibleBlock\(in_channels=6\) for _ in range\(num_blocks\)\]\)', init_code, content)

# Update extract method
extract_old = """        # 3. EXACT DEMODULATION
        # Recompute the mask from the watermarked image to perfectly reverse the modulation
        mask = self.saliency(watermarked).detach()
        demodulated_secret = raw_secret / mask

        return torch.clamp(self.enhance_post(demodulated_secret), -1.0, 1.0)"""

extract_new = """        # 3. LEARNED DEMODULATION
        # Recompute the mask from the watermarked image to guide demodulation
        mask = self.saliency(watermarked).detach()
        demodulated_secret = self.demodulator(raw_secret, mask)

        return torch.clamp(self.enhance_post(demodulated_secret), -1.0, 1.0)"""

content = content.replace(extract_old, extract_new)

with open("models/generator.py", "w") as f:
    f.write(content)
