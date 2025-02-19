from typing import Tuple, List
import numpy as np
import os
from skimage.io import imread
from skimage.transform import resize
from pathlib import Path
from torch.utils.data import Dataset
import torch


class Crack(Dataset):

    def __init__(self, data_root: Path, input_frames: int = 12, seq_len: int = 20,
                 image_size: Tuple[int, int] = (256, 132)):
        self.crack_video_frames: List[str] = sorted(os.listdir(data_root))
        self.data_root: Path = data_root
        self.input_frames: int = input_frames
        self.seq_len: int = seq_len
        self.image_size: Tuple[int, int] = image_size
        self.mean = 0
        self.std = 1

    def __len__(self):
        return len(self.crack_video_frames)

    def __getitem__(self, index):
        image_frames = []

        for frame_idx, img_path in enumerate(sorted((Path(self.data_root) /
                                                     self.crack_video_frames[index]).glob('*.jpg'),
                                                    key=lambda file: int(file.name.split('_')[-1].split('.')[0]))):
            image_frame = imread(str(img_path))
            image_frame_resized = resize(image_frame, output_shape=self.image_size, preserve_range=True) / 255.0
            image_frames.append(image_frame_resized)

        image_frames_sub_sampled = self._subsample_images(image_frames[52:])
        inputs = image_frames_sub_sampled[:self.input_frames]
        outputs = image_frames_sub_sampled[self.input_frames:self.seq_len]
        inputs = torch.from_numpy(inputs).contiguous().float()
        outputs = torch.from_numpy(outputs).contiguous().float()
        inputs = inputs.transpose(2, 3).transpose(1, 2)
        outputs = outputs.transpose(2, 3).transpose(1, 2)
        return inputs, outputs

    def _subsample_images(self, image_frames: List[np.ndarray]) -> np.ndarray:
        image_frames_np = np.zeros((self.seq_len, self.image_size[0], self.image_size[1], 3),
                                   dtype=np.float32)
        stride = int(np.ceil(len(image_frames) / self.seq_len))
        image_frames_sub = image_frames[::stride]
        assert len(image_frames_sub) == self.seq_len, f"After subsampling length of image frame should be equal to " \
                                                      f"the sequence length got image_frames={len(image_frames_sub)}" \
                                                      f" for seq_len={self.seq_len}"
        for img_idx, img in enumerate(image_frames_sub):
            image_frames_np[img_idx] = img
        return image_frames_np


def load_data(batch_size: int, val_batch_size: int, num_workers: int, data_root: Path, train_dir: Path, val_dir: Path):
    train_data = Crack(
        data_root=train_dir,
        input_frames=8,
        seq_len=16,
        image_size=(256, 128))
    test_data = Crack(
        data_root=val_dir,
        input_frames=8,
        seq_len=16,
        image_size=(256, 128))
    dataloader_train = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    dataloader_test = torch.utils.data.DataLoader(
        test_data, batch_size=val_batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
    return dataloader_train, None, dataloader_test, 0, 1
