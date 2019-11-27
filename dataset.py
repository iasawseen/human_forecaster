import torch
import torch.utils.data


# Disable multiprocesing for numpy/opencv. We already multiprocess ourselves, this would mean every subprocess produces
# even more threads which would lead to a lot of context switching, slowing things down a lot.

import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np

import warnings
warnings.filterwarnings("ignore")

from generate import ANNS_TRAIN_FILE_PATH, ANNS_VAL_FILE_PATH
from utils import load


class PoseDataset(torch.utils.data.Dataset):
    def __init__(self, data_type='train', seq_len=12, future=0, start_at_center=False, velocity=False):
        if data_type == 'train':
            self.data = load(ANNS_TRAIN_FILE_PATH)
        else:
            self.data = load(ANNS_VAL_FILE_PATH)
        self.data = [np.array(data) for data in self.data]

        self.seq_len = seq_len
        self.future = future
        self.start_at_center = start_at_center
        self.velocity = velocity

        self.inputs = list()
        self.targets = list()

        for sequence in self.data:
            for i in range(sequence.shape[0] - self.seq_len - self.future):
                self.inputs.append(sequence[i: i + self.seq_len, :].astype(np.float32))
                self.targets.append(sequence[i + self.seq_len + self.future, :].astype(np.float32))

    def __len__(self):
        # return len(self.data)
        return len(self.inputs)

    def __getitem__(self, idx):
        return {'input': torch.from_numpy(self.inputs[idx]), 'target': torch.from_numpy(self.targets[idx])}

    # def __getitem__(self, idx):
    #     data_sequence = np.array(self.data[idx])
    #     if self.velocity:
    #         data_sequence[:-1] -= data_sequence[1:]
    #
    #     data_sequence = data_sequence[:-1]
    #
    #     length = data_sequence.shape[0]
    #
    #     max_index = length - self.seq_len - self.future
    #     start_index = np.random.randint(0, max_index)
    #
    #     input_data = data_sequence[start_index: start_index + self.seq_len, :].astype(np.float32)
    #
    #     if self.start_at_center:
    #         input_data -= input_data[0, :]
    #
    #     target_data = data_sequence[start_index + self.seq_len + self.future, :].astype(np.float32)
    #
    #     return {'input': torch.from_numpy(input_data), 'target': torch.from_numpy(target_data)}


def get_dataloaders(batch_size=32, seq_len=12, future=12):
    train_dataset = PoseDataset('train', seq_len=seq_len, future=future)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size, shuffle=True, num_workers=4
    )

    val_dataset = PoseDataset('val', seq_len=seq_len, future=future)

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size, shuffle=False, num_workers=4
    )

    return train_dataloader, val_dataloader
