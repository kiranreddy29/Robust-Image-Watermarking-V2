import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from models.generator import WatermarkGenerator
from noise_layers.Gaussian_noise import Gaussian_Noise
from noise_layers.jpeg_compression import JpegCompression
from noise_layers.quantization import Quantization
from noise_layers.cutout import CutoutAttack
from utils.dataset import get_loader

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    G = WatermarkGenerator(num_blocks=8).to(device)

    model_path = "generator_final.pth"
    if not os.path.exists(model_path):
        print("Model not found. Run training first.")
        return

    G.load_state_dict(torch.load(model_path, map_location=device))
    G.eval()

    cover_loader = get_loader("data/DIV2K/cover", batch_size=1)
    wm_loader    = get_loader("data/DIV2K/watermark", batch_size=1)

    cover = next(iter(cover_loader)).to(device)
    secret = next(iter(wm_loader)).to(device)

    attacks = {
        "Gaussian 1":  Gaussian_Noise(mean=0.0, sigma=1.0/127.5),
        "Gaussian 10": Gaussian_Noise(mean=0.0, sigma=10.0/127.5),
        "JPEG Q=80":   JpegCompression(device=device, yuv_keep_weights=(15, 5, 5)),
        "Round":       Quantization(device=device),
        "Cutout":      CutoutAttack(drop_prob=0.1, block_size=40)
    }

    with torch.no_grad():
        watermarked = G.embed(cover, secret)

        fig, axes = plt.subplots(4, len(attacks), figsize=(18, 12))

        for i, (name, attack) in enumerate(attacks.items()):
            attacked = attack([watermarked.clone(), cover.clone()])[0]
            extracted = G.extract(attacked, watermarked)

            axes[0, i].imshow((watermarked[0].cpu().permute(1,2,0) * 0.5 + 0.5).clamp(0,1))
            axes[0, i].set_title(name, fontsize=14, fontweight='bold')
            if i == 0:
                axes[0, i].set_ylabel("Watermarked Image", fontsize=12)

            axes[1, i].imshow((secret[0].cpu().permute(1,2,0) * 0.5 + 0.5).clamp(0,1))
            if i == 0:
                axes[1, i].set_ylabel("Original Secret", fontsize=12)

            axes[2, i].imshow((extracted[0].cpu().permute(1,2,0) * 0.5 + 0.5).clamp(0,1))
            if i == 0:
                axes[2, i].set_ylabel("Extracted Secret", fontsize=12)

            diff = torch.abs(secret - extracted) * 10
            axes[3, i].imshow(diff[0].cpu().permute(1,2,0).clamp(0,1))
            if i == 0:
                axes[3, i].set_ylabel("Difference (10x)", fontsize=12)

            for row in range(4):
                axes[row, i].set_xticks([])
                axes[row, i].set_yticks([])

        plt.tight_layout()
        plt.savefig("visual_comparison_report.png", dpi=300)
        print("Saved visual_comparison_report.png")

if __name__ == "__main__":
    main()