import numpy as np
import torch

def longest_center_band(mask_arr):

    # 첫 번째 1의 인덱스와 두 번째 1의 인덱스 찾기
    first_one_idx = np.where(mask_arr == 1)[0][0]
    last_one_idx = np.where(mask_arr == 1)[0][1]
    gap = last_one_idx - first_one_idx

    center_idx = len(mask_arr) // 2
    center_band_start = None
    center_band_end = None

    for i in range(len(mask_arr)//2 - 1):
        if mask_arr[center_idx - i] == 0 and center_band_start is None:
            center_band_start = center_idx - i
        if mask_arr[center_idx + i] == 0 and center_band_end is None:
            center_band_end = center_idx + i

        if center_band_start is not None and center_band_end is not None:
            break


    for i in range(first_one_idx, len(mask_arr), gap):
        if i == center_band_start:
            center_band_start += 1
        if i == center_band_end:
            center_band_end -= 1

    return center_band_start, center_band_end, gap

def kspace_augmentation(mask, acc, seed=13):
    """
    Apply k-space augmentation based on the acceleration factor.
    mask: (length, 0)
    """
    np.random.seed(seed)

    new_mask = np.zeros_like(mask, dtype=np.float32)

    s, e, gap = longest_center_band(mask)

    # ACS 영역 넓히거나 좁힘
    # center_add = np.random.choice([-1, 0, 1])

    # s = s + center_add
    # e = e - center_add

    for i in range(len(mask)):
        if s <= i <= e:
            new_mask[i] = 1.0
        else:
            new_mask[i] = 0.0

    if acc == 4:
        # 0, 1, 2, 3 중에 시작 위치 랜덤 선택
        start = np.random.randint(0, 4)
        # 4칸씩 건너뛰기
        for i in range(start, len(new_mask), 4):
            new_mask[i] = 1.0

    elif acc == 8:
        # 0, 1, 2, 3, 4, 5, 6, 7 중에 시작 위치 랜덤 선택
        start = np.random.randint(0, 8)
        # 8칸씩 건너뛰기
        for i in range(start, len(new_mask), 8):
            new_mask[i] = 1.0

    else:
        start = np.random.randint(0, gap)
        for i in range(start, len(new_mask), gap):
            new_mask[i] = 1.0

    return new_mask

def to_tensor(data):
    """
    Convert numpy array to PyTorch tensor. For complex arrays, the real and imaginary parts
    are stacked along the last dimension.
    Args:
        data (np.array): Input numpy array
    Returns:
        torch.Tensor: PyTorch version of data
    """
    return torch.from_numpy(data)

class DataTransform:
    def __init__(self, isforward, max_key):
        self.isforward = isforward
        self.max_key = max_key
    def __call__(self, mask, input, input_img, target, attrs, fname, slice, kspace_augment, acc):
        if not self.isforward:
            target = to_tensor(target)
            maximum = attrs[self.max_key]
        else:
            target = -1
            maximum = -1

        if kspace_augment:
            mask = kspace_augmentation(mask, acc)

        
        kspace = to_tensor(input * mask)
        kspace = torch.stack((kspace.real, kspace.imag), dim=-1)
        mask = torch.from_numpy(mask.reshape(1, 1, kspace.shape[-2], 1).astype(np.float32)).byte()
        return mask, kspace, input_img, target, maximum, fname, slice
