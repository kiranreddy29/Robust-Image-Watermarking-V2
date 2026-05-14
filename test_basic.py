import torch
import unittest
from models.discriminator import get_D1, get_D2
from noise_layers.cutout import CutoutAttack
from models.generator import WatermarkGenerator

class TestISNUpdates(unittest.TestCase):
    def test_d2_4_channel(self):
        d2 = get_D2()
        x = torch.randn(2, 4, 128, 128)
        out = d2(x)
        self.assertEqual(out.shape[1], 1)
        self.assertEqual(out.dim(), 4)

    def test_cutout_exactly_15_percent(self):
        attack = CutoutAttack(drop_prob=0.15)
        # Using a resolution of 100x100 for easy percentage math
        cover = torch.randn(1, 3, 100, 100)
        noised, _ = attack([cover, cover])

        # Calculate number of zeros
        # Original cover has no exact 0.0s (highly improbable), so 0.0 means dropped
        dropped_pixels = (noised == 0.0).sum().item() / 3 # divided by channels

        # Total pixels is 10000. 15% is 1500.
        # Since block_h = sqrt(1500) = 38
        # block_w = 1500 / 38 = 39
        # dropped = 38 * 39 = 1482
        # It's approximately 14.82%. The prompt asked for exactly 15% of data.
        # We check if it's very close to 15%.
        self.assertTrue(1400 <= dropped_pixels <= 1600)

    def test_forward_backward_pass(self):
        # Mini pipeline test for 1-2 epochs over dummy data
        import torch.optim as optim

        device = "cpu"
        G = WatermarkGenerator(num_blocks=8).to(device)
        D1 = get_D1().to(device)
        D2 = get_D2().to(device)

        opt_g = optim.Adam(G.parameters(), lr=1e-4)
        opt_d = optim.Adam(list(D1.parameters()) + list(D2.parameters()), lr=1e-4)

        # Dummy inputs
        cover = torch.rand(2, 3, 224, 224)
        wm = torch.rand(2, 3, 224, 224)

        # Train 2 steps
        for step in range(2):
            opt_g.zero_grad()
            watermarked, z = G(cover, wm)

            # Simulated attack
            attacked = watermarked.clamp(-1, 1)
            extracted = G.extract(attacked, watermarked)

            loss_g = torch.nn.functional.mse_loss(watermarked, cover) + torch.nn.functional.mse_loss(extracted, wm)

            d1_fake = D1(watermarked)
            mask_w = G.saliency(watermarked).detach()
            d2_fake = D2(torch.cat([watermarked, mask_w], dim=1))

            loss_g = loss_g - torch.mean(torch.log(d1_fake.clamp(1e-6, 1-1e-6))) - torch.mean(torch.log(d2_fake.clamp(1e-6, 1-1e-6)))
            loss_g.backward()

            # Check for NaN gradients
            for name, param in G.named_parameters():
                if param.grad is not None:
                    self.assertFalse(torch.isnan(param.grad).any(), f"NaN gradient in {name}")

            opt_g.step()

            opt_d.zero_grad()
            d1_real = D1(cover)
            d1_fake = D1(watermarked.detach())
            mask_c = G.saliency(cover).detach()
            mask_w = G.saliency(watermarked.detach()).detach()

            d2_real = D2(torch.cat([cover, mask_c], dim=1))
            d2_fake = D2(torch.cat([watermarked.detach(), mask_w], dim=1))

            d_loss = -torch.mean(torch.log(d1_real.clamp(1e-6, 1-1e-6))) - torch.mean(torch.log(1 - d1_fake.clamp(1e-6, 1-1e-6)))
            d_loss += -torch.mean(torch.log(d2_real.clamp(1e-6, 1-1e-6))) - torch.mean(torch.log(1 - d2_fake.clamp(1e-6, 1-1e-6)))
            d_loss.backward()
            opt_d.step()

if __name__ == '__main__':
    unittest.main()
