import torch
import numpy as np
import cv2
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from mmdet.apis import init_detector, inference_detector
from collections import defaultdict


def get_box_center(box):
    return (box[1][0] + box[0][0]) // 2, (box[1][1] + box[0][1]) // 2


class PoseDetector:
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

    @staticmethod
    def convert_bgr_to_rgb(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def predict(self, img):
        img = self.convert_bgr_to_rgb(img) / 255
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

        predictions = [
            {
                'anchor': get_box_center(box),
                'box': box,
                'point_of_interest': (
                    np.array(prediction_dict['left_shoulder'][index]) +
                    np.array(prediction_dict['right_shoulder'][index])
                ) / 2
            }
            for index, box in enumerate(prediction_dict['boxes'])
        ]

        return predictions


class HeadDetector:
    def __init__(self, cfg, score_threshold=0.75, nms_threshold=0.75, cuda_id=0):
        self.cfg = cfg
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold

        torch.cuda.set_device(cuda_id)

        config_file = self.cfg.DETECTING.HEAD_CONFIG_FILE_PATH
        checkpoint_file = self.cfg.DETECTING.HEAD_CHECKPOINT_FILE_PATH

        self.model = init_detector(
            config_file, checkpoint_file,
            device='cuda:{}'.format(cuda_id)
        )

        if 'retinanet' in config_file:
            self.model.cfg['test_cfg']['score_thr'] = score_threshold
            self.model.cfg['test_cfg']['nms']['iou_thr'] = nms_threshold
        else:
            self.model.cfg['test_cfg']['rcnn']['score_thr'] = score_threshold
            self.model.cfg['test_cfg']['rcnn']['nms']['iou_thr'] = nms_threshold

    def predict(self, img):
        result = inference_detector(self.model, img)

        bboxes = np.vstack(result)

        prediction_dict = defaultdict(list)

        for box in bboxes:
            x_min, y_min = box[:2]
            x_max, y_max = box[2:4]

            prediction_dict['boxes'].append(((x_min, y_min), (x_max, y_max)))
            prediction_dict['scores'].append(box[4])

        predictions = [
            {
                'anchor': get_box_center(box),
                'box': box,
                'point_of_interest': get_box_center(box)
            }
            for index, box in enumerate(prediction_dict['boxes'])
        ]

        return predictions
