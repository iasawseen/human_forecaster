import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from utils import save, load

PARENT_FOLDER = '/home/marcus/data/sber/data/'

ANN_FOLDER_TRAIN = PARENT_FOLDER + 'posetrack_data/annotations/train/'
ANN_FOLDER_VAL = PARENT_FOLDER + 'posetrack_data/annotations/val/'

ANNS_TRAIN_FILE_PATH = PARENT_FOLDER + 'train_anns.pkl'
ANNS_VAL_FILE_PATH = PARENT_FOLDER + 'val_anns.pkl'

DATASET_DIR = '/home/marcus/data/sber/data/MOT16/train/'

TRAIN_DIRS = (
    'MOT16-02/', 'MOT16-04/',
    # 'MOT16-05/',
    # 'MOT16-10/',
    # 'MOT16-11/',
    # 'MOT16-13/',
)
VAL_DIRS = ('MOT16-09/',)

DATA_PATH = 'gt/gt.txt'


def extract_data(file_path, min_length=32):
    with open(file_path, 'r') as f:
        datas = [[float(el) for el in line.strip().split(',')] for line in f.readlines()]

    for data in datas:
        for i in range(4):
            data[i] = int(data[i])

    track_dict = defaultdict(list)

    for data in datas:
        track_dict[data[1]].append(list(data))

    def extract_trajectory(sequence):
        return [data[2:4] for data in sequence]

    for track_id in track_dict:
        track_dict[track_id] = np.array(extract_trajectory(track_dict[track_id]))

    track_dict_filtered = {
        track_id: track_dict[track_id] for track_id in track_dict if track_dict[track_id].shape[0] > min_length
    }

    return track_dict_filtered


def extract_dirs(dirs, min_length=32):
    trajectories = list()

    for directory in dirs:
        dir_trajectories = extract_data(DATASET_DIR + directory + DATA_PATH, min_length=min_length)
        for track_id in dir_trajectories:
            trajectories.append(dir_trajectories[track_id])

    return trajectories


if __name__ == '__main__':
    train_trajectories = extract_dirs(TRAIN_DIRS, min_length=128)
    val_trajectories = extract_dirs(VAL_DIRS, min_length=128)

    save(train_trajectories, file_path=ANNS_TRAIN_FILE_PATH)
    save(val_trajectories, file_path=ANNS_VAL_FILE_PATH)
