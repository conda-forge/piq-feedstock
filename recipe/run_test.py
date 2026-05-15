import torch
import piq


image = torch.rand(1, 3, 32, 32)
perturbed = torch.clamp(image * 0.9 + 0.05, 0.0, 1.0)

psnr = piq.psnr(image, perturbed, data_range=1.0)
ssim = piq.ssim(image, perturbed, data_range=1.0)

assert torch.isfinite(psnr)
assert psnr.item() > 20.0
assert 0.0 < ssim.item() <= 1.0
