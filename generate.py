from ipywidgets import Video, Image
from IPython.display import display
import numpy as np
import cv2
import json
import base64
from collections import Counter
import matplotlib.pyplot as plt
from utils import save, load
from collections import defaultdict
import os

PARENT_FOLDER = '/home/marcus/data/sber/data/'

ANN_FOLDER_TRAIN = PARENT_FOLDER + 'posetrack_data/annotations/train/'
ANN_FOLDER_VAL = PARENT_FOLDER + 'posetrack_data/annotations/val/'

ANNS_TRAIN_FILE_PATH = PARENT_FOLDER + 'train_anns.pkl'
ANNS_VAL_FILE_PATH = PARENT_FOLDER + 'val_anns.pkl'


def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


def generate_ann_dicts(annotation):
    id_to_img = {img_ann['frame_id']: img_ann for img_ann in annotation['images']}
    id_to_ann = {img_ann['image_id']: img_ann for img_ann in annotation['annotations']}

    return id_to_img, id_to_ann


def generate_track_sequencies(annotations):
    track_id_to_sequence = defaultdict(list)

    for ann in annotations:
        track_id = ann['track_id']
        track_id_to_sequence[track_id].append(ann)

    return track_id_to_sequence


def process_track_sequencies(track_to_sequence):
    for track in track_to_sequence:
        track_to_sequence[track] = [el['keypoints'][:2] for el in track_to_sequence[track]]

    return track_to_sequence


def generate_data(ann_folder, min_length=30):
    anns = [os.path.join(ann_folder, file_path) for file_path in os.listdir(ann_folder)]
    anns = [load_json(ann_file_path) for ann_file_path in anns]
    anns = [generate_track_sequencies(ann['annotations']) for ann in anns]
    anns = [process_track_sequencies(track_to_sequence) for track_to_sequence in anns]

    processed_anns = list()

    for sequences in anns:
        for track in sequences:
            if len(sequences[track]) >= min_length:
                processed_anns.append(sequences[track])

    return processed_anns


if __name__ == '__main__':
    train_anns = generate_data(ANN_FOLDER_TRAIN)
    val_anns = generate_data(ANN_FOLDER_VAL)

    save(train_anns, file_path=ANNS_TRAIN_FILE_PATH)
    save(val_anns, file_path=ANNS_VAL_FILE_PATH)
