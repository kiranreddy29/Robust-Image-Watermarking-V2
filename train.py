import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips
import random

from torch.optim.lr_scheduler import MultiStepLR
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from utils.metrics import psnr
from noise_layers.cutout import CutoutAttack
from models.generator import WatermarkGenerator
from models.discriminator import Discriminator
from noise_layers.noiser import Noiser
from noise_layers.Gaussian_noise import Gaussian_Noise
from noise_layers.identity import Identity
from noise_layers.jpeg_compression import JpegCompression
from noise_layers.quantization import Quantization
from utils.dataset import get_loader

torch.autograd.set_detect_anomaly(False)
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def haar_dwt2d(x):
    x00 = x[:, :, 0::2, 0::2]; x10 = x[:, :, 1::2, 0::2]
    x01 = x[:, :, 0::2, 1::2]; x11 = x[:, :, 1::2, 1::2]
    ll = (x00 + x10 + x01 + x11) / 2.0
    lh = (x00 - x10 + x01 - x11) / 2.0
    hl = (x00 + x10 - x01 - x11) / 2.0
    hh = (x00 - x10 - x01 + x11) / 2.0
    return ll, lh, hl, hh


def dwt_loss(x, y):
    """FIX: Emphasize low-freq (visible) more, ignore high-freq mismatch."""
    llx, lhx, hlx, hhx = haar_dwt2d(x)
    lly, lhy, hly, hhy = haar_dwt2d(y)
    return (F.l1_loss(llx, lly)
            + 0.1 * (F.l1_loss(lhx, lhy) + F.l1_loss(hlx, hly) + F.l1_loss(hhx, hhy)))


def discriminator_loss(real, fake):
    real = torch.clamp(real, 1e-6, 1 - 1e-6)
    fake = torch.clamp(fake, 1e-6, 1 - 1e-6)
    return -(torch.mean(torch.log(real)) + torch.mean(torch.log(1.0 - fake)))


def generator_adv_loss(fake):
    fake = torch.clamp(fake, 1e-6, 1 - 1e-6)
    return -torch.mean(torch.log(fake))


def safe_lpips(loss_fn, a, b, device):
    try:
        val = loss_fn(a.float(), b.float()).mean()
        if torch.isnan(val) or torch.isinf(val):
            return torch.tensor(0.0, device=device)
        return val
    except Exception:
        return torch.tensor(0.0, device=device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--phase1_epoch', type=int, default=30)   # NEW: clean training
    parser.add_argument('--phase2_epoch', type=int, default=80)   # adversarial
    parser.add_argument('--phase3_epoch', type=int, default=160)  # LR drop
    parser.add_argument('--lr_early', type=float, default=2e-4)
    parser.add_argument('--lr_late', type=float, default=2e-5)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--resume_epoch', type=int, default=0)    # FIX: default 0
    parser.add_argument('--composed_attacks', action='store_true')
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    G = WatermarkGenerator(num_blocks=8).to(device)
    D = Discriminator().to(device)

    start_epoch = 0
    if args.resume_epoch > 0:
        resume_path = os.path.join(args.checkpoint_dir, f"ckpt_epoch{args.resume_epoch}.pth")
        if os.path.exists(resume_path):
            print("Loading checkpoint:", resume_path)
            ckpt = torch.load(resume_path, map_location=device)
            G.load_state_dict(ckpt["G"], strict=False)  # strict=False due to arch change
            start_epoch = ckpt["epoch"]
            print(f"Resumed from epoch {start_epoch}")

    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        G = nn.DataParallel(G)
        D = nn.DataParallel(D)

    loss_fn_lpips = lpips.LPIPS(net='alex').to(device)
    loss_fn_lpips.eval()

    # FIX: identity layer weighted higher by appearing multiple times
    noise_list = [
        Identity(), Identity(),  # 2x weight on clean
        Gaussian_Noise(mean=0.0, sigma=5.0 / 127.5),
        Gaussian_Noise(mean=0.0, sigma=10.0 / 127.5),
        JpegCompression(device=device),
        Quantization(device=device),
        CutoutAttack(drop_prob=0.10, block_size=32),
    ]
    attack_module = Noiser(noise_list, composed=args.composed_attacks).to(device)
    identity_only = Noiser([Identity()]).to(device)

    mse = nn.MSELoss()
    l1  = nn.L1Loss()

    opt_g = torch.optim.Adam(G.parameters(), lr=args.lr_early, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=args.lr_early, betas=(0.5, 0.999))

    gamma = args.lr_late / args.lr_early
    sch_g = MultiStepLR(opt_g, milestones=[args.phase3_epoch], gamma=gamma)
    sch_d = MultiStepLR(opt_d, milestones=[args.phase3_epoch], gamma=gamma)

    for _ in range(start_epoch):
        sch_g.step()
        if start_epoch >= args.phase2_epoch:
            sch_d.step()

    scaler = GradScaler("cuda")
    cover_loader = get_loader("/kaggle/input/datasets/sharansmenon/div2k/DIV2K_train_HR/DIV2K_train_HR",  batch_size=args.batch_size, shuffle=True)
    wm_loader    = get_loader("/kaggle/input/datasets/sharansmenon/div2k/DIV2K_valid_HR/DIV2K_valid_HR", batch_size=args.batch_size, shuffle=True)

    print("Starting Training...")

    for epoch in range(start_epoch, args.epochs):
        G.train(); D.train()
        epoch_g = 0.0; epoch_d = 0.0
        pc_total = 0.0; ps_total = 0.0; n = 0

        # FIX: Curriculum learning - no attacks in Phase 1
        if epoch < args.phase1_epoch:
            current_attack = identity_only
            lambda_secret = 2.0       # heavily emphasize secret first
            lambda_cover  = 1.0
        elif epoch < args.phase2_epoch:
            current_attack = attack_module
            lambda_secret = 1.5
            lambda_cover  = 1.0
        else:
            current_attack = attack_module
            lambda_secret = 1.0
            lambda_cover  = 1.5

        pbar = tqdm(zip(cover_loader, wm_loader),
                    total=min(len(cover_loader), len(wm_loader)),
                    desc=f"Epoch {epoch+1}/{args.epochs}")

        for cover, wm in pbar:
            if isinstance(cover, (list, tuple)): cover = cover[0]
            if isinstance(wm, (list, tuple)):    wm = wm[0]
            b_size = min(cover.size(0), wm.size(0))
            cover = cover.to(device, non_blocking=True)
            wm    = wm.to(device, non_blocking=True)

            opt_g.zero_grad(set_to_none=True)

            with autocast("cuda"):
                watermarked, z_latent = G(cover, wm)

                # FIX: Don't clamp during training — use straight-through estimator
                attacked = current_attack([watermarked, cover])[0]
                # Straight-through clamp (identity gradient)
                attacked_clamped = attacked.detach().clamp(-1, 1) + (attacked - attacked.detach())

                if isinstance(G, nn.DataParallel):
                    extracted = G.module.extract(attacked_clamped, watermarked)
                else:
                    extracted = G.extract(attacked_clamped, watermarked)

                # Cover losses
                L_cover_mse = mse(watermarked, cover)
                L_cover_l1  = l1(watermarked, cover)
                L_dwt       = dwt_loss(watermarked, cover)
                L_lpips     = safe_lpips(loss_fn_lpips, watermarked, cover, device)

                # Secret losses
                L_secret_mse = mse(extracted, wm)
                L_secret_l1  = l1(extracted, wm)

                # FIX: Weak z regularization (encourages but doesn't force z→0)
                L_z = torch.mean(z_latent ** 2)

                cover_loss  = L_cover_mse + 0.5 * L_cover_l1 + 0.3 * L_dwt + 0.1 * L_lpips
                secret_loss = L_secret_mse + 0.5 * L_secret_l1

                # FIX: z weight reduced from 1.0 → 0.001
                g_loss = lambda_cover * cover_loss + lambda_secret * secret_loss + 0.0001 * L_z

                if epoch >= args.phase2_epoch:
                    d_fake_g = D(watermarked.float())
                    # FIX: bumped adv weight from 0.01 → 0.05
                    g_loss = g_loss + 0.125 * generator_adv_loss(d_fake_g)
            scaler.scale(g_loss).backward()
            scaler.unscale_(opt_g)
            torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0)
            scaler.step(opt_g)
            scaler.update()

            d_loss_val = 0.0
            if epoch >= args.phase2_epoch:
                opt_d.zero_grad(set_to_none=True)
                real_img = cover.detach().float()
                fake_img = watermarked.detach().float()
                d_real = D(real_img)
                d_fake = D(fake_img)
                d_loss = discriminator_loss(d_real, d_fake)
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(D.parameters(), 1.0)
                opt_d.step()
                d_loss_val = d_loss.item()

            with torch.no_grad():
                pc = psnr(watermarked.float(), cover.float(), mode='cover')
                ps = psnr(extracted.float(), wm.float(), mode='secret')

            epoch_g += g_loss.item(); epoch_d += d_loss_val
            pc_total += pc; ps_total += ps; n += 1

            pbar.set_postfix({"G": f"{g_loss.item():.3f}", "D": f"{d_loss_val:.3f}",
                              "PC": f"{pc:.1f}", "PS": f"{ps:.1f}"})

        sch_g.step()
        if epoch >= args.phase2_epoch:
            sch_d.step()

        avg_pc = pc_total / n; avg_ps = ps_total / n
        print(f"Epoch [{epoch+1}/{args.epochs}] "
              f"G:{epoch_g/n:.4f} D:{epoch_d/n:.4f} "
              f"PC:{avg_pc:.2f} PS:{avg_ps:.2f}")

        if (epoch + 1) % 10 == 0:
            save_G = G.module.state_dict() if isinstance(G, nn.DataParallel) else G.state_dict()
            torch.save({"epoch": epoch + 1, "G": save_G,
                        "avg_psnr_c": avg_pc, "avg_psnr_s": avg_ps},
                       os.path.join(args.checkpoint_dir, f"ckpt_epoch{epoch+1}.pth"))
            print("Checkpoint saved.")

    save_G = G.module.state_dict() if isinstance(G, nn.DataParallel) else G.state_dict()
    torch.save(save_G, "generator_final.pth")
    print("Training Complete.")


if __name__ == "__main__":
    main()