import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    """
    Read a grayscale sequence image, returning (T,1,H,W), with the label being the last mask (1,H,W).
    If resize_to_256=True, scale uniformly to 256×256.
    """

    def __init__(self,
                 data_root: str,
                 sequence_length: int = 5,
                 sequence_prefix: str = "Sequence",
                 img_dir: str = "images",
                 mask_dir: str = "masks",
                 resize_to_256: bool = False):
        super().__init__()
        self.data_root = data_root
        self.seq_len = sequence_length
        self.seq_prefix = sequence_prefix
        self.img_dir_name = img_dir
        self.mask_dir_name = mask_dir
        self.resize_to_256 = resize_to_256

        self.samples = []   # (sequence_folder, start_idx)

        # Scan all sequential folders
        for item in os.listdir(data_root):
            seq_path = os.path.join(data_root, item)
            if not (os.path.isdir(seq_path) and item.startswith(self.seq_prefix)):
                continue

            img_folder = os.path.join(seq_path, self.img_dir_name)
            mask_folder = os.path.join(seq_path, self.mask_dir_name)
            if not (os.path.isdir(img_folder) and os.path.isdir(mask_folder)):
                continue

            img_names = sorted([f for f in os.listdir(img_folder)
                                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
            mask_names = sorted([f for f in os.listdir(mask_folder)
                                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])

            if len(img_names) != len(mask_names) or len(img_names) < self.seq_len:
                continue

            for start_idx in range(0, len(img_names) - self.seq_len + 1, 3):
                self.samples.append((seq_path, start_idx, img_names, mask_names))

        if len(self.samples) == 0:
            raise RuntimeError("No valid sequences found.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq_path, start_idx, img_names, mask_names = self.samples[idx]

        imgs = []
        for i in range(self.seq_len):
            frame_idx = start_idx + i
            img_path = os.path.join(seq_path, self.img_dir_name, img_names[frame_idx])
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise IOError(f"Cannot read image {img_path}")
            if self.resize_to_256:
                img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
            img = img.astype(np.float32) / 255.0
            imgs.append(img[np.newaxis, ...])          # (1,H,W)

        img_seq = np.concatenate(imgs, axis=0)         # (T,1,H,W)

        mask_idx = start_idx + self.seq_len - 1
        mask_path = os.path.join(seq_path, self.mask_dir_name, mask_names[mask_idx])
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        # print("Mask foreground ratio:", mask.sum() / mask.size)
        if mask is None:
            raise IOError(f"Cannot read mask {mask_path}")
        if self.resize_to_256:
            mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype(np.float32)
        mask = mask[np.newaxis, ...]                   # (1,H,W)

        img_seq = torch.from_numpy(img_seq)
        mask = torch.from_numpy(mask)

        img_seq = torch.stack([img_seq], dim=0)
        mask = torch.stack([mask], dim=0)

        return img_seq, mask


class TestSequenceDataset(Dataset):
    def __init__(self, seq_root: str, seq_len: int = 5, resize_to_256: bool = False):
        self.seq_len = seq_len
        self.resize = resize_to_256
        img_dir = os.path.join(seq_root, 'images')
        mask_dir = os.path.join(seq_root, 'masks')
        self.img_paths = sorted([os.path.join(img_dir, f)
                                 for f in os.listdir(img_dir)
                                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
        self.mask_paths = sorted([os.path.join(mask_dir, f)
                                  for f in os.listdir(mask_dir)
                                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
        if len(self.img_paths) != len(self.mask_paths):
            raise RuntimeError('images/masks count mismatch')
        if len(self.img_paths) < seq_len:
            raise RuntimeError('sequence too short')

    def __len__(self):
        return len(self.img_paths) - self.seq_len + 1

    def __getitem__(self, idx):
        imgs = []
        for i in range(self.seq_len):
            img = cv2.imread(self.img_paths[idx + i], cv2.IMREAD_GRAYSCALE)
            if self.resize:
                img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
            img = img.astype(np.float32) / 255.0
            imgs.append(img[np.newaxis, ...])     # (1,H,W)
        img_seq = np.concatenate(imgs, axis=0)    # (10,1,H,W) -> (10,H,W)

        mask = cv2.imread(self.mask_paths[idx + self.seq_len - 1], cv2.IMREAD_GRAYSCALE)
        if self.resize:
            mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype(np.float32)[np.newaxis, ...]   # (1,H,W)

        return torch.from_numpy(img_seq).unsqueeze(0), \
               torch.from_numpy(mask).unsqueeze(0), \
               self.mask_paths[idx + self.seq_len - 1]  # Used to save the file name
