import torch
import numpy as np
from utils import load
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from collections import defaultdict
from sklearn.linear_model import HuberRegressor, LinearRegression
from mmdet.apis import init_detector, inference_detector, show_result, show_result_pyplot
import mmcv


# DETECTOR_CONFIG = '/home/marcus/libs/mmdetection/configs/head_retinanet_r50_fpn_1x.py'
# DETECTOR_CHECKPOINT = '/home/marcus/libs/mmdetection/work_dirs/retinanet_r50_fpn_1x/epoch_7.pth'

DETECTOR_CONFIG = '/home/marcus/libs/mmdetection/configs/head_cascade_rcnn_dconv_c3-c5_r50_fpn_1x.py'
DETECTOR_CHECKPOINT = '/home/marcus/libs/mmdetection/work_dirs/head_cascade/epoch_11.pth'


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


class MMDetector:
    def __init__(self, score_threshold=0.75, nms_threshold=0.75, cuda_id=0):
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold

        torch.cuda.set_device(cuda_id)

        config_file = DETECTOR_CONFIG
        checkpoint_file = DETECTOR_CHECKPOINT

        self.model = init_detector(config_file, checkpoint_file, device='cuda:{}'.format(cuda_id))

        if 'retinanet' in config_file:
            self.model.cfg['test_cfg']['score_thr'] = score_threshold
            self.model.cfg['test_cfg']['nms']['iou_thr'] = nms_threshold
        else:
            self.model.cfg['test_cfg']['rcnn']['score_thr'] = score_threshold
            self.model.cfg['test_cfg']['rcnn']['nms']['iou_thr'] = nms_threshold

    def predict(self, img):
        result = inference_detector(self.model, img)

        bboxes = np.vstack(result)

        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(result)
        ]

        labels = np.concatenate(labels)

        prediction_dict = defaultdict(list)

        for box in bboxes:
            # if box[4] > self.score_threshold:
            x_min, y_min = box[:2]
            x_max, y_max = box[2:4]

            prediction_dict['boxes'].append(((x_min, y_min), (x_max, y_max)))
            prediction_dict['scores'].append(box[4])

        return prediction_dict
