import h5py
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

class MRISliceImageDataset(Dataset):
    def __init__(self, data_dir):
        # Only search for h5 files under the 'image' subdirectory
        image_dir = Path(data_dir) / 'image'
        self.files = list(image_dir.glob('*.h5'))
        self.samples = []
        for f in self.files:
            with h5py.File(f, 'r') as hf:
                if 'image_input' in hf:
                    num_slices = hf['image_input'].shape[0]
                else:
                    continue
            fname_lower = f.name.lower()
            if 'brain' in fname_lower:
                label = 0
            elif 'knee' in fname_lower:
                label = 1
            else:
                continue
            for i in range(num_slices):
                self.samples.append((f, i, label))
        # Debug info: file count, sample count, class distribution
        from collections import Counter
        label_list = [lbl for _, _, lbl in self.samples]
        print(f"[DEBUG] {data_dir} -> Found {len(self.files)} h5 files, {len(self.samples)} slices. Class dist: {Counter(label_list)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        f, slice_idx, label = self.samples[idx]
        with h5py.File(f, 'r') as hf:
            img = hf['image_input'][slice_idx]
            image = torch.tensor(img, dtype=torch.float32)
            if image.ndim == 2:
                image = image.unsqueeze(0)
        return image, label

def load_image_data(data_dir, batch_size=8, shuffle=False):
    dataset = MRISliceImageDataset(data_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader
