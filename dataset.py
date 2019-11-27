import torch
import torch.utils.data

import os
# os.environ["OMP_NUM_THREADS"] = "1"
from generate import ANNS_TRAIN_FILE_PATH, ANNS_VAL_FILE_PATH
from utils import load
import numpy as np

import warnings
warnings.filterwarnings("ignore")


class PoseDataset(torch.utils.data.Dataset):
    def __init__(self, data_type='train', seq_len=12, future=0):
        self.data_type = data_type

        if self.data_type == 'train':
            self.data = load(ANNS_TRAIN_FILE_PATH)
        else:
            self.data = load(ANNS_VAL_FILE_PATH)

        self.data = [np.array(data) for data in self.data]

        self.seq_len = seq_len
        self.future = future

        self.inputs = list()
        self.targets = list()

        for sequence in self.data:
            for i in range(sequence.shape[0] - self.seq_len - self.future):
                input = sequence[i: i + self.seq_len, :].astype(np.float32)
                target = sequence[i + self.seq_len: i + self.seq_len + self.future, :].astype(np.float32)

                x_min = min(np.min(input[:, 0]), np.min(target[:, 0]))
                y_min = min(np.min(input[:, 1]), np.min(target[:, 1]))

                if x_min < 0:
                    input[:, 0] -= x_min
                    target[:, 0] -= x_min

                if y_min < 0:
                    input[:, 1] -= y_min
                    target[:, 1] -= y_min

                self.inputs.append(input)
                self.targets.append(target)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input = np.array(self.inputs[idx])
        target = np.array(self.targets[idx])

        if self.data_type == 'train' and np.random.rand() > 0.5:
            input[:, 0] = 1920 - input[:, 0]
            target[:, 0] = 1920 - target[:, 0]

        # if self.data_type == 'train' and np.random.rand() > 0.5:
        #     input[:, 1] = 1080 - input[:, 1]
        #     target[1] = 1080 - target[1]

        if self.data_type == 'train' and np.random.rand() > 0.5:
            x_min = min(np.min(input[:, 0]), np.min(target[:, 0]))
            x_max = max(np.max(input[:, 0]), np.max(target[:, 0]))

            y_min = min(np.min(input[:, 1]), np.min(target[:, 1]))
            y_max = max(np.max(input[:, 1]), np.max(target[:, 1]))

            x_half_range = min(x_min, 1920 - x_max, 0)
            y_half_range = y_min

            x_shift = np.random.randint(-x_half_range, x_half_range + 1)
            # y_shift = np.random.randint(-y_half_range, y_half_range + 1)

            try:
                y_shift = np.random.randint(-y_half_range, y_half_range + 1)
            except ValueError:
                print(-y_half_range, y_half_range + 1)
                print(y_min)
                exit()

            input[:, 0] += x_shift
            target[:, 0] += x_shift

            input[:, 1] += y_shift
            target[:, 1] += y_shift

        assert(min(np.min(input[:, 0]), np.min(target[:, 0])) >= 0)
        assert(min(np.min(input[:, 1]), np.min(target[:, 1])) >= 0)

        return {'input': torch.from_numpy(input), 'target': torch.from_numpy(target.flatten())}


# class PoseDataset(torch.utils.data.Dataset):
#     def __init__(self, data_type='train', seq_len=12, future=0):
#         self.data_type = data_type
#
#         if self.data_type == 'train':
#             self.data = load(ANNS_TRAIN_FILE_PATH)
#         else:
#             self.data = load(ANNS_VAL_FILE_PATH)
#
#         self.data = [np.array(data) for data in self.data]
#
#         self.seq_len = seq_len
#         self.future = future
#
#         self.inputs = list()
#         self.targets = list()
#
#         for sequence in self.data:
#             for i in range(sequence.shape[0] - self.seq_len - self.future):
#                 input = sequence[i: i + self.seq_len, :].astype(np.float32)
#                 target = sequence[i + self.seq_len + self.future, :].astype(np.float32)
#
#                 x_min = min(np.min(input[:, 0]), np.min(target[0]))
#                 y_min = min(np.min(input[:, 1]), np.min(target[1]))
#
#                 if x_min < 0:
#                     input[:, 0] -= x_min
#                     target[0] -= x_min
#
#                 if y_min < 0:
#                     input[:, 1] -= y_min
#                     target[1] -= y_min
#
#                 self.inputs.append(input)
#                 self.targets.append(target)
#
#     def __len__(self):
#         return len(self.inputs)
#
#     def __getitem__(self, idx):
#         input = np.array(self.inputs[idx])
#         target = np.array(self.targets[idx])
#
#         if self.data_type == 'train' and np.random.rand() > 0.5:
#             input[:, 0] = 1920 - input[:, 0]
#             target[0] = 1920 - target[0]
#
#         # if self.data_type == 'train' and np.random.rand() > 0.5:
#         #     input[:, 1] = 1080 - input[:, 1]
#         #     target[1] = 1080 - target[1]
#
#         if self.data_type == 'train' and np.random.rand() > 0.5:
#             x_min = min(np.min(input[:, 0]), np.min(target[0]))
#             x_max = max(np.max(input[:, 0]), np.max(target[0]))
#
#             y_min = min(np.min(input[:, 1]), np.min(target[1]))
#             y_max = max(np.max(input[:, 1]), np.max(target[1]))
#
#             x_half_range = min(x_min, 1920 - x_max, 0)
#             y_half_range = y_min
#
#             x_shift = np.random.randint(-x_half_range, x_half_range + 1)
#             # y_shift = np.random.randint(-y_half_range, y_half_range + 1)
#
#             try:
#                 y_shift = np.random.randint(-y_half_range, y_half_range + 1)
#             except ValueError:
#                 print(-y_half_range, y_half_range + 1)
#                 print(y_min)
#                 exit()
#
#             input[:, 0] += x_shift
#             target[0] += x_shift
#
#             input[:, 1] += y_shift
#             target[1] += y_shift
#
#         assert(min(np.min(input[:, 0]), np.min(target[0])) >= 0)
#         assert(min(np.min(input[:, 1]), np.min(target[1])) >= 0)
#
#         return {'input': torch.from_numpy(input), 'target': torch.from_numpy(target)}


def get_dataloaders(batch_size=32, seq_len=12, future=12):
    train_dataset = PoseDataset('train', seq_len=seq_len, future=future)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size, shuffle=True, num_workers=0
    )

    val_dataset = PoseDataset('val', seq_len=seq_len, future=future)

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size, shuffle=False, num_workers=0
    )

    return train_dataloader, val_dataloader
