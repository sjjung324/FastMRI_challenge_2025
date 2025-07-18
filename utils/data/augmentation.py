import numpy as np
import torch
import torch.nn.functional as F
from utils.model.fastmri import fft2c, ifft2c, rss, complex_abs
from utils.common.utils import center_crop
import matplotlib.pyplot as plt

# --- Complex-valued transforms ---
def complex_flip(img, horizontal=True):
    # img: (S, C, H, W, 2) complex coil images
    if horizontal:
        return img.flip(-2)
    else:
        return img.flip(-3)

def complex_translate(img, max_pixels=8):
    # Integer translation
    tx = np.random.randint(-max_pixels, max_pixels+1)
    ty = np.random.randint(-max_pixels, max_pixels+1)
    return torch.roll(img, shifts=(ty, tx), dims=(-3, -2))

def complex_affine(img, rot_range=10, scale_range=0.1, shear_range=5):
    # Affine transform with bicubic interpolation
    angle = np.random.uniform(-rot_range, rot_range)
    scale = np.random.uniform(1-scale_range, 1+scale_range)
    shear = np.random.uniform(-shear_range, shear_range)
    theta = torch.tensor([
        [scale * np.cos(np.deg2rad(angle)), -np.sin(np.deg2rad(angle+shear)), 0],
        [np.sin(np.deg2rad(angle)), scale * np.cos(np.deg2rad(angle+shear)), 0]
    ], dtype=torch.float)
    S, C, H, W, _ = img.shape
    out = []
    for s in range(S):
        coil_out = []
        for c in range(C):
            real = img[s, c, :, :, 0].unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            imag = img[s, c, :, :, 1].unsqueeze(0).unsqueeze(0)
            grid = F.affine_grid(theta.unsqueeze(0), real.shape, align_corners=False).to(real.device)
            real_t = F.grid_sample(real, grid, mode='bicubic', align_corners=False).squeeze(0).squeeze(0)
            imag_t = F.grid_sample(imag, grid, mode='bicubic', align_corners=False).squeeze(0).squeeze(0)
            coil_out.append(torch.stack([real_t, imag_t], dim=-1))  # (H, W, 2)
        out.append(torch.stack(coil_out, dim=0))  # (C, H, W, 2)
    return torch.stack(out, dim=0)  # (S, C, H, W, 2)


def augment_kspace(kspace, augment_config=None):
    """
    kspace: (S, C, H, W, 2) complex-valued multi-coil k-space tensor
    augment_config: dict with keys for each transform and their parameters
    Returns: augmented k-space tensor (S, C, H, W, 2) and corresponding image domain tensor (S, C, H, W, 2)
    """
    augment_config = augment_config or {}
    # 1. kspace -> image domain
    img = ifft2c(kspace) # (S, C, H, W, 2)
    # 2. Apply augmentations (no RSS)
    if augment_config.get('flip', False):
        img = complex_flip(img, horizontal=augment_config.get('flip_horizontal', True))
    if augment_config.get('translate', False):
        img = complex_translate(img, max_pixels=augment_config.get('translate_max', 8))
    if augment_config.get('affine', False):
        img = complex_affine(
            img,
            rot_range=augment_config.get('affine_rot', 10),
            scale_range=augment_config.get('affine_scale', 0.1),
            shear_range=augment_config.get('affine_shear', 5)
        )
    # 3. image domain -> kspace
    kspace_aug = fft2c(img)
    target = center_crop(rss(complex_abs(img), dim=1), 384, 384)  # (S, 384, 384)
    return kspace_aug, target

def visualize_augmentation(kspace, augment_config=None, coil_idx=0):
    """
    kspace: (S, C, H, W, 2) complex-valued multi-coil k-space tensor
    augment_config: dict for augmentation
    coil_idx: which coil to plot (int or 'all')
    """
    # 1. 원본/augment된 image (배치별)
    img_orig = rss(complex_abs(ifft2c(kspace)), dim=1)  # (S, H, W)
    img_orig = center_crop(img_orig, 384, 384)
    kspace_aug, _ = augment_kspace(kspace, augment_config)
    img_aug = rss(complex_abs(ifft2c(kspace_aug)), dim=1)  # (S, H, W)
    img_aug = center_crop(img_aug, 384, 384)

    S = img_orig.shape[0]
    # coil_idx는 C축 기준, S개 배치 모두 시각화
    plt.figure(figsize=(8, 4 * S))
    for s in range(S):
        plt.subplot(S, 2, 2*s+1)
        plt.imshow(img_orig[s].cpu(), cmap='gray')
        plt.title(f'Original Sample {s}')
        plt.axis('off')
        plt.subplot(S, 2, 2*s+2)
        plt.imshow(img_aug[s].cpu(), cmap='gray')
        plt.title(f'Augmented Sample {s}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()