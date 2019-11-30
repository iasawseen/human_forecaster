import torch
import numpy as np
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from collections import defaultdict


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
