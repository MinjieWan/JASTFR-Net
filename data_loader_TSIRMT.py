import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    """
    TSIRMT版本
    读取灰度序列图，返回 (1,T,1,H,W)，标签为最后一张掩膜(1,1,H,W)
    """

    def __init__(self,
                 data_root: str,
                 mode: str = 'train',
                 sequence_length: int = 5,
                 img_dir: str = "images",
                 mask_dir: str = "masks",
                 resize_to_256: bool = False):
        super().__init__()

        self.data_root = os.path.join(data_root, mode)
        self.seq_len = sequence_length
        self.img_dir_name = img_dir
        self.mask_dir_name = mask_dir
        self.resize_to_256 = resize_to_256

        self.samples = []

        img_root = os.path.join(self.data_root, self.img_dir_name)
        mask_root = os.path.join(self.data_root, self.mask_dir_name)

        # ✅ 扫描所有序列文件夹
        sequences = sorted(os.listdir(img_root))

        for seq in sequences:
            img_folder = os.path.join(img_root, seq)
            mask_folder = os.path.join(mask_root, seq)

            if not (os.path.isdir(img_folder) and os.path.isdir(mask_folder)):
                continue

            img_names = sorted(
                [f for f in os.listdir(img_folder)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))],
                key=self._sort_key
            )

            mask_names = sorted(
                [f for f in os.listdir(mask_folder)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))],
                key=self._sort_key
            )

            if len(img_names) != len(mask_names) or len(img_names) < self.seq_len:
                continue

            for start_idx in range(0, len(img_names) - self.seq_len + 1, 1):
                self.samples.append((img_folder, mask_folder, start_idx, img_names, mask_names))

        if len(self.samples) == 0:
            raise RuntimeError("No valid sequences found.")

        print(f"[{mode}] Total samples: {len(self.samples)}")

    def _sort_key(self, name):
        base = os.path.splitext(name)[0]
        return int(base)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_folder, mask_folder, start_idx, img_names, mask_names = self.samples[idx]

        imgs = []
        for i in range(self.seq_len):
            frame_idx = start_idx + i
            img_path = os.path.join(img_folder, img_names[frame_idx])

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise IOError(f"Cannot read image {img_path}")

            if self.resize_to_256:
                img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)

            img = img.astype(np.float32) / 255.0
            imgs.append(img[np.newaxis, ...])

        img_seq = np.concatenate(imgs, axis=0)   # (T,1,H,W)

        mask_idx = start_idx + self.seq_len - 1
        mask_path = os.path.join(mask_folder, mask_names[mask_idx])

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise IOError(f"Cannot read mask {mask_path}")

        if self.resize_to_256:
            mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)

        mask = (mask > 0).astype(np.float32)
        mask = mask[np.newaxis, ...]

        img_seq = torch.from_numpy(img_seq)
        mask = torch.from_numpy(mask)

        img_seq = torch.stack([img_seq], dim=0)
        mask = torch.stack([mask], dim=0)

        return img_seq, mask


class TestSequenceDataset(Dataset):
    def __init__(self, seq_dir: str, seq_len: int = 5, resize_to_256: bool = False):
        self.seq_len = seq_len
        self.resize = resize_to_256

        self.seq_name = os.path.basename(seq_dir)
        self.root = os.path.dirname(os.path.dirname(seq_dir))

        img_dir = seq_dir
        mask_dir = os.path.join(self.root, 'masks', self.seq_name)

        if not os.path.isdir(img_dir):
            raise RuntimeError(f'Image folder not found: {img_dir}')
        if not os.path.isdir(mask_dir):
            raise RuntimeError(f'Mask folder not found: {mask_dir}')

        self.img_paths = sorted(
            [os.path.join(img_dir, f)
             for f in os.listdir(img_dir)
             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))],
            key=self._sort_key
        )

        self.mask_paths = sorted(
            [os.path.join(mask_dir, f)
             for f in os.listdir(mask_dir)
             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))],
            key=self._sort_key
        )

        if len(self.img_paths) != len(self.mask_paths):
            raise RuntimeError('images/masks count mismatch')

        if len(self.img_paths) < self.seq_len:
            raise RuntimeError('sequence too short')

    def _sort_key(self, path):
        name = os.path.basename(path)
        base = os.path.splitext(name)[0]
        return int(base)

    def __len__(self):
        return len(self.img_paths) - self.seq_len + 1

    def __getitem__(self, idx):
        imgs = []

        for i in range(self.seq_len):
            img = cv2.imread(self.img_paths[idx + i], cv2.IMREAD_GRAYSCALE)

            if self.resize:
                img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)

            img = img.astype(np.float32) / 255.0
            imgs.append(img[np.newaxis, ...])

        img_seq = np.concatenate(imgs, axis=0)

        mask = cv2.imread(self.mask_paths[idx + self.seq_len - 1], cv2.IMREAD_GRAYSCALE)

        if self.resize:
            mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)

        mask = (mask > 0).astype(np.float32)[np.newaxis, ...]

        return torch.from_numpy(img_seq).unsqueeze(0), \
               torch.from_numpy(mask).unsqueeze(0), \
               self.mask_paths[idx + self.seq_len - 1]
