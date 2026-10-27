"""Masked photometric and scale-aware camera-Z losses."""

import torch
import torch.nn.functional as F


def masked_ssim(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    """Mean SSIM over complete 11×11 valid windows; no invalid-pixel leakage."""
    x = prediction.permute(2, 0, 1)[None]
    y = target.permute(2, 0, 1)[None]
    axis = torch.arange(11, dtype=x.dtype, device=x.device) - 5
    gaussian = torch.exp(-(axis**2) / (2 * 1.5**2))
    gaussian /= gaussian.sum()
    kernel = (gaussian[:, None] * gaussian[None, :])[None, None].expand(3, 1, 11, 11)

    def blur(value):
        return F.conv2d(value, kernel, groups=3)

    mean_x, mean_y = blur(x), blur(y)
    var_x = (blur(x * x) - mean_x.square()).clamp_min(0)
    var_y = (blur(y * y) - mean_y.square()).clamp_min(0)
    covariance = blur(x * y) - mean_x * mean_y
    numerator = (2 * mean_x * mean_y + 0.01**2) * (2 * covariance + 0.03**2)
    denominator = (mean_x.square() + mean_y.square() + 0.01**2) * (var_x + var_y + 0.03**2)
    score = (numerator / denominator).mean(dim=1)[0]
    counts = F.conv2d(mask[None, None].to(x.dtype), x.new_ones((1, 1, 11, 11)))[0, 0]
    windows = counts >= 121
    if not windows.any():
        # Mask boundaries can leave no complete window. Keep a useful RGB gradient.
        return 1 - (prediction - target).abs()[mask].mean()
    return score[windows].mean()


def rgb_depth_loss(rendered, rgb, depth, image_mask, geometry_mask, ssim_weight, depth_weight):
    l1 = (rendered.rgb - rgb).abs()[image_mask].mean()
    ssim = masked_ssim(rendered.rgb, rgb, image_mask)
    # Invalid map pixels can be NaN; sanitize BEFORE arithmetic to keep backward finite.
    safe_target = torch.where(geometry_mask, depth, torch.ones_like(depth))
    relative = (rendered.depth - safe_target).abs() / safe_target.clamp_min(1e-8)
    depth_loss = relative[geometry_mask].mean()
    photometric = (1 - ssim_weight) * l1 + ssim_weight * (1 - ssim)
    total = photometric + depth_weight * depth_loss
    return total, {"l1": l1, "ssim": ssim, "relative_depth": depth_loss}
