import torch
import math

def psnr(img1, img2, mode='cover'):
    mse = torch.mean((img1 - img2) ** 2).item()
    if mse < 1e-10: return 100.0
    
    # Standard PSNR formula. Max signal for [-1, 1] range is 2.0
    raw_db = 20.0 * math.log10(2.0 / math.sqrt(mse))
    return raw_db