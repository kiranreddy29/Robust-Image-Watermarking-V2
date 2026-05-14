import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        # PatchGAN Architecture
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, 4, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(512, 1, 4, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

def get_D1():
    """Global Realism: 3-input-channel PatchGAN"""
    return Discriminator(in_channels=3)

def get_D2():
    """Edge Consistency: 4-input-channel PatchGAN"""
    return Discriminator(in_channels=4)
