import torch
import numpy as np
from utils import load
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from collections import defaultdict
from sklearn.linear_model import HuberRegressor, LinearRegression


class Detector:
    COCO_PERSON_KEYPOINT_NAMES = [
        'nose',
        'left_eye',
        'right_eye',
        'left_ear',
        'right_ear',
        'left_shoulder',
        'right_shoulder',
        'left_elbow',
        'right_elbow',
        'left_wrist',
        'right_wrist',
        'left_hip',
        'right_hip',
        'left_knee',
        'right_knee',
        'left_ankle',
        'right_ankle'
    ]

    def __init__(self, score_threshold=0.5, nms_threshold=0.5):
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.model = keypointrcnn_resnet50_fpn(
            pretrained=True,
            box_score_thresh=self.score_threshold,
            box_nms_thresh=self.nms_threshold
        ).cuda()
        self.model.eval()
        self.name_to_index = {name: index for index, name in enumerate(self.COCO_PERSON_KEYPOINT_NAMES)}

    def get_coors(self, keypoint, name):
        return keypoint[self.name_to_index[name], 0], keypoint[self.name_to_index[name], 1]

    def predict(self, img):
        img = img.transpose((2, 0, 1))
        x = [torch.from_numpy(img.astype(np.float32)).cuda()]

        with torch.no_grad():
            [prediction] = self.model(x)

        boxes = prediction['boxes'].cpu().numpy()
        scores = prediction['scores'].cpu().numpy()
        keypoints = prediction['keypoints'].cpu().numpy()

        prediction_dict = defaultdict(list)

        for box, score, keypoint in zip(boxes, scores, keypoints):
            box = box.astype(int)
            keypoint = keypoint.astype(int)

            x_min, y_min = box[:2]
            x_max, y_max = box[2:]

            prediction_dict['boxes'].append(((x_min, y_min), (x_max, y_max)))

            for keypoint_name in self.COCO_PERSON_KEYPOINT_NAMES:
                prediction_dict[keypoint_name].append(self.get_coors(keypoint, keypoint_name))

        return prediction_dict


class ModelWrapper:
    def __init__(self, file_path_pattern, length):
        self.model = load(file_path=file_path_pattern.format(length))
        self.length = length

    def process_chunk(self, chunk):
        def get_axis_features(chunk, axis):
            axis_min, axis_max, axis_mean = chunk[:, axis].min(), chunk[:, axis].max(), chunk[:, axis].mean()
            axis_median, axis_std = np.median(chunk[:, axis]), chunk[:, axis].std()

            return [
                axis_min, axis_max, axis_mean, axis_median, axis_std,
            ]

        output = chunk.flatten().tolist()
        output.extend(get_axis_features(chunk, axis=0))
        output.extend(get_axis_features(chunk, axis=1))

        return np.array(output)

    def predict(self, queue):
        def smooth_data(traj):
            x_train, y_train = traj[:, 0].reshape((-1, 1)), traj[:, 1]
            huber = LinearRegression()
            huber.fit(x_train, y_train)
            prediction = huber.predict(x_train).reshape((-1, 1))
            return np.hstack((x_train, prediction))

        data = np.array(queue)
        data = smooth_data(data)
        data = data[-self.length:]

        data = self.process_chunk(data).reshape((1, -1))
        prediction = self.model.predict(data).flatten()[:2]
        return prediction
