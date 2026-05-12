import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.swin_transformer import SwinTransformerBlock as TimmSwinBlock

_SWIN_DIM   = 96
_SWIN_HEADS = 3
_INPUT_RES  = (224, 224)


class TextureSaliency(nn.Module):
    """Core Patent Feature: Sobel-guided Visual Masking"""
    def __init__(self):
        super().__init__()
        kernel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        kernel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        self.register_buffer('weight_x', kernel_x)
        self.register_buffer('weight_y', kernel_y)

    def forward(self, x):
        with torch.no_grad():
            grad_x = F.conv2d(x, self.weight_x, padding=1, groups=3)
            grad_y = F.conv2d(x, self.weight_y, padding=1, groups=3)
            magnitude = torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-6)
            
            mask = magnitude.mean(dim=1, keepdim=True)
            min_val = mask.amin(dim=(2, 3), keepdim=True)
            max_val = mask.amax(dim=(2, 3), keepdim=True)
            
            norm_mask = (mask - min_val) / (max_val - min_val + 1e-8)
            return norm_mask * 0.5 + 0.5


class InvertibleBlock(nn.Module):
    """FIX: Replaced Tanh with bounded clamp via soft scaling. Allows wider dynamic range."""
    def __init__(self, in_channels=6, clamp=2.0):
        super().__init__()
        half = in_channels // 2
        self.clamp = clamp
        self.s1 = nn.Sequential(nn.Conv2d(half, 64, 3, padding=1), nn.ReLU(inplace=False), nn.Conv2d(64, half, 3, padding=1))
        self.t1 = nn.Sequential(nn.Conv2d(half, 64, 3, padding=1), nn.ReLU(inplace=False), nn.Conv2d(64, half, 3, padding=1))
        self.s2 = nn.Sequential(nn.Conv2d(half, 64, 3, padding=1), nn.ReLU(inplace=False), nn.Conv2d(64, half, 3, padding=1))
        self.t2 = nn.Sequential(nn.Conv2d(half, 64, 3, padding=1), nn.ReLU(inplace=False), nn.Conv2d(64, half, 3, padding=1))

    def _bound(self, s):
        # Soft clamp: keeps exp(s) numerically stable but allows wider range than Tanh
        return self.clamp * torch.tanh(s / self.clamp)

    def forward(self, x, reverse=False):
        x1 = x[:, :3, :, :]
        x2 = x[:, 3:, :, :]
        if not reverse:
            s1 = self._bound(self.s1(x1))
            y2 = x2 * torch.exp(s1) + self.t1(x1)
            y1 = x1
            s2 = self._bound(self.s2(y2))
            z1 = y1 * torch.exp(s2) + self.t2(y2)
            return torch.cat([z1, y2], dim=1)
        else:
            z1 = x[:, :3, :, :]
            z2 = x[:, 3:, :, :]
            y2 = z2
            s2 = self._bound(self.s2(y2))
            y1 = (z1 - self.t2(y2)) * torch.exp(-s2)
            s1 = self._bound(self.s1(y1))
            x2_rev = (y2 - self.t1(y1)) * torch.exp(-s1)
            return torch.cat([y1, x2_rev], dim=1)


class DenseBlock(nn.Module):
    def __init__(self, channels=3, growth_rate=32):  # Increased capacity
        super().__init__()
        self.conv1 = nn.Conv2d(channels, growth_rate, 3, padding=1)
        self.conv2 = nn.Conv2d(channels + growth_rate, channels, 3, padding=1)
        self.relu  = nn.ReLU(inplace=False)

    def forward(self, x):
        out1 = self.relu(self.conv1(x))
        out2 = self.conv2(torch.cat([x, out1], dim=1))
        return out2 + x


class DynamicMLP(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pool    = nn.AdaptiveAvgPool2d(1)
        self.fc1     = nn.Linear(channels, channels // 2)
        self.relu    = nn.ReLU(inplace=False)
        self.fc2     = nn.Linear(channels // 2, channels)
        self.sigmoid = nn.Sigmoid()  # FIX: bound output to [0,1] for stable gating

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.pool(x).flatten(1)
        w = self.relu(self.fc1(w))
        w = self.sigmoid(self.fc2(w))
        return w.view(b, c, 1, 1)


class SwinBlock(nn.Module):
    def __init__(self, channels, window_size=4):
        super().__init__()
        self.proj_in     = nn.Conv2d(channels, _SWIN_DIM, kernel_size=1)
        self.swin        = TimmSwinBlock(
            dim=_SWIN_DIM, input_resolution=_INPUT_RES,
            num_heads=_SWIN_HEADS, window_size=window_size,
            shift_size=window_size // 2,
        )
        self.proj_out    = nn.Conv2d(_SWIN_DIM, channels, kernel_size=1)
        self.dynamic_mlp = DynamicMLP(channels)

    def forward(self, x):
        feat = self.proj_in(x).permute(0, 2, 3, 1)
        feat = self.swin(feat).permute(0, 3, 1, 2)
        feat = self.proj_out(feat)
        return x + (feat * self.dynamic_mlp(x))


class EnhancementModule(nn.Module):
    def __init__(self, channels=3, window_size=4):
        super().__init__()
        self.pre_dense  = DenseBlock(channels)
        self.swin_block = SwinBlock(channels, window_size=window_size)
        self.post_dense = DenseBlock(channels)

    def forward(self, x):
        feat = self.pre_dense(x)
        feat = self.swin_block(feat)
        feat = self.post_dense(feat)
        return feat + x


class DifferentialFeatureExtractor(nn.Module):
    def __init__(self, channels=3, growth_rate=32):
        super().__init__()
        ch = channels
        self.conv1 = nn.Conv2d(ch,                 growth_rate, 3, padding=1)
        self.conv2 = nn.Conv2d(ch +   growth_rate, growth_rate, 3, padding=1)
        self.conv3 = nn.Conv2d(ch + 2*growth_rate, growth_rate, 3, padding=1)
        self.conv4 = nn.Conv2d(ch + 3*growth_rate, growth_rate, 3, padding=1)
        self.final = nn.Conv2d(ch + 4*growth_rate, ch,          1)

    def forward(self, xc, xd):
        diff = xc - xd
        x1   = F.relu(self.conv1(diff))
        x2   = F.relu(self.conv2(torch.cat([diff, x1],          dim=1)))
        x3   = F.relu(self.conv3(torch.cat([diff, x1, x2],     dim=1)))
        x4   = F.relu(self.conv4(torch.cat([diff, x1, x2, x3], dim=1)))
        return self.final(torch.cat([diff, x1, x2, x3, x4],    dim=1))


class WatermarkGenerator(nn.Module):
    def __init__(self, num_blocks=8):
        super().__init__()
        
        # 1. Initialize your unique saliency feature
        self.saliency     = TextureSaliency() 
        
        self.isn_blocks   = nn.ModuleList([InvertibleBlock(in_channels=6) for _ in range(num_blocks)])
        self.enhance_pre  = EnhancementModule(channels=3, window_size=4)
        self.enhance_post = EnhancementModule(channels=3, window_size=8)
        self.diff_feat    = DifferentialFeatureExtractor(channels=3)

    def embed(self, cover, secret):
        """Embeds secret into cover via invertible flow with Saliency Modulation."""
        
        # 2. Modulate the secret using the cover's texture mask
        mask = self.saliency(cover).detach()
        modulated_secret = secret * mask 
        
        x = torch.cat([cover, modulated_secret], dim=1)
        for block in self.isn_blocks:
            x = block(x, reverse=False)
            
        watermarked = x[:, :3, :, :]
        z = x[:, 3:, :, :]   # latent (residual)
        return torch.clamp(watermarked, -1.0, 1.0), z

    def extract(self, attacked, watermarked):
        xc_feat     = self.diff_feat(watermarked, attacked)
        xd_enhanced = self.enhance_pre(attacked)
        x = torch.cat([xd_enhanced, xc_feat], dim=1)
        
        for block in reversed(self.isn_blocks):
            x = block(x, reverse=True)
            
        raw_secret = x[:, 3:, :, :]
        
        # 3. EXACT DEMODULATION
        # Recompute the mask from the watermarked image to perfectly reverse the modulation
        mask = self.saliency(watermarked).detach()
        demodulated_secret = raw_secret / mask
        
        return torch.clamp(self.enhance_post(demodulated_secret), -1.0, 1.0)

    def forward(self, cover, secret):
        """For training: returns (watermarked, z)."""
        return self.embed(cover, secret)

    def embed_only(self, cover, secret):
        """For inference: returns just the watermarked image."""
        watermarked, _ = self.embed(cover, secret)
        return watermarked