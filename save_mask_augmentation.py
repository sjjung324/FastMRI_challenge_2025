import h5py
import numpy as np
from pathlib import Path

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

    return center_band_start, center_band_end, gap, first_one_idx

def kspace_augmentation(mask):
    """
    Apply k-space augmentation based on the acceleration factor.
    mask: (length, 0)
    """
    acs_mask = np.zeros_like(mask, dtype=np.float32)

    s, e, gap, first_one_idx = longest_center_band(mask)

    # ACS 영역 넓히거나 좁힘
    # center_add = np.random.choice([-1, 0, 1])

    # s = s + center_add
    # e = e - center_add

    # ACS 영역 1로 채우기
    for i in range(len(mask)):
        if s <= i <= e:
            acs_mask[i] = 1.0
        else:
            acs_mask[i] = 0.0

    
    new_fist_one_idxs = [i for i in range(gap) if i != first_one_idx]
    aug_masks = []

    for new_first_one_idx in new_fist_one_idxs:
        new_mask = acs_mask.copy()
        for i in range(new_first_one_idx, len(mask), gap):
            new_mask[i] = 1.0
        aug_masks.append(new_mask)

    return aug_masks


def save_augmented_masks(folder_path):
    """
    For each h5 file in folder_path, replace ['mask'] with augmented masks and save as new files.
    Each augmented mask is saved as a new h5 file with suffix '_aug{i}.h5'.
    """
    folder = Path(folder_path)
    out_folder = folder / 'augmented_masks'
    out_folder.mkdir(exist_ok=True)
    mask_files = sorted(folder.glob('*.h5'))
    for mask_file in mask_files:
        with h5py.File(mask_file, 'r') as f:
            mask = np.asarray(f['mask']).astype(np.float32)
            # Copy all other datasets/attributes
            data_dict = {k: np.array(f[k]) for k in f.keys() if k != 'mask'}
        aug_masks = kspace_augmentation(mask)
        for i, aug_mask in enumerate(aug_masks):
            aug_file = out_folder / (mask_file.stem + f'_aug{i}.h5')
            with h5py.File(aug_file, 'w') as f_out:
                # Save augmented mask
                f_out.create_dataset('mask', data=aug_mask.astype(np.float32))
                # Save other datasets
                for k, v in data_dict.items():
                    f_out.create_dataset(k, data=v)
            print(f'Saved augmented mask to {aug_file}')


if __name__ == "__main__":
    train_folder_path = Path("C:/fast_mri/Data/train/kspace/")
    
    save_augmented_masks(train_folder_path)