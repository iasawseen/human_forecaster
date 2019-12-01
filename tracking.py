from scipy.optimize import linear_sum_assignment
import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
import seaborn as sns
import itertools
import copy
from model import HeadDetector, PoseDetector
from typing import List, Dict, Tuple


class KalmanWrapper:
    def __init__(self, coors: Tuple, state_noise: float = 1.0, r_scale: float = 1.0, q_var: float = 1.0):
        self.filter = KalmanFilter(dim_x=4, dim_z=2)
        self.filter.x = np.array([coors[0], 0, coors[1], 0])

        self.dt = 1.0

        self.filter.F = np.array([[1, self.dt, 0, 0],
                                  [0, 1, 0, 0],
                                  [0, 0, 1, self.dt],
                                  [0, 0, 0, 1]])

        self.filter.H = np.array([[1, 0, 0, 0],
                                  [0, 0, 1, 0]])

        self.filter.P *= state_noise
        self.filter.R = np.diag(np.ones(2)) * state_noise * r_scale
        self.filter.Q = Q_discrete_white_noise(dim=2, dt=self.dt, var=q_var, block_size=2)

    def update(self, coors):
        self.filter.update(coors)

    def predict(self):
        self.filter.predict()

    def get_coors(self) -> Tuple:
        return self.filter.x[0], self.filter.x[2]


class KalmanTracker:
    def __init__(self,
                 detection: Dict,
                 state_noise: float = 1.0,
                 r_scale: float = 1.0,
                 q_var: float = 1.0,
                 color: Tuple = (255, 255, 255)
                 ):
        self.color = color
        self.anchor_filter = KalmanWrapper(
            detection['anchor'],
            state_noise=state_noise, r_scale=r_scale, q_var=q_var
        )

        self.point_of_interest_filter = KalmanWrapper(
            detection['point_of_interest'],
            state_noise=state_noise, r_scale=r_scale, q_var=q_var
        )
        self.box_size = self.box_to_size(detection['box'])

        self.hits = 0
        self.misses = 0
        self.age = 1

    @staticmethod
    def box_to_size(box: Tuple[Tuple]) -> Tuple:
        return box[1][0] - box[0][0], box[1][1] - box[0][1]

    def update(self, detection: Dict):
        self.anchor_filter.predict()
        self.anchor_filter.update(detection['anchor'])
        self.point_of_interest_filter.predict()
        self.point_of_interest_filter.update(detection['point_of_interest'])
        self.box_size = self.box_to_size(detection['box'])
        self.age += 1
        self.hits += 1
        self.misses = 0

    def update_with_estimation(self):
        self.anchor_filter.predict()
        self.point_of_interest_filter.predict()
        self.age += 1
        self.misses += 1

    def get_state(self) -> Dict:
        return {
            'anchor': self.anchor_filter.get_coors(),
            'box_size': self.get_box_size(),
            'point_of_interest': self.point_of_interest_filter.get_coors(),
            'color': self.color
        }

    def get_estimation(self) -> Tuple:
        return self.anchor_filter.get_coors()

    def get_box_size(self) -> Tuple:
        return self.box_size

    def get_misses(self) -> int:
        return self.misses

    def get_hits(self) -> int:
        return self.hits


class Tracking:
    def __init__(self, cfg, state_noise=40.0, r_scale=10.0, q_var=1.0, iou_threshold=0.3, max_misses=4, min_hits=1):
        self.cfg = cfg

        if self.cfg.MAIN.HEAD_DETECTION:
            self.detector = HeadDetector(
                self.cfg,
                score_threshold=cfg.DETECTING.SCORE_THRESHOLD,
                nms_threshold=cfg.DETECTING.NMS_IOU_THRESHOLD
            )
        else:
            self.detector = PoseDetector(
                score_threshold=cfg.DETECTING.SCORE_THRESHOLD,
                nms_threshold=cfg.DETECTING.NMS_IOU_THRESHOLD
            )

        self.state_noise = state_noise
        self.r_scale = r_scale
        self.q_var = q_var

        self.iou_threshold = iou_threshold
        self.frame_count = 0
        self.max_misses = max_misses
        self.min_hits = min_hits
        self.trackers = list()
        self.tracker_palette = itertools.cycle(sns.color_palette())

    def track(self, next_frame: np.array) -> List[Dict]:
        self.frame_count += 1

        detections = self.detector.predict(next_frame)

        if len(self.trackers) == 0 and len(detections) == 0:
            return list()

        self._update_trackers(detections)

        # filter trackers
        self.trackers = [tracker for tracker in self.trackers if tracker.get_misses() <= self.max_misses]

        filtered_detections = [tracker.get_state() for tracker in self.trackers if tracker.get_hits() >= self.min_hits]

        return filtered_detections

    def predict_trajectories(self, future_len: int = 32, min_age: int = 6) -> Tuple[List, List]:
        # copy trackers in order to preserve states of original trackers
        trackers = [copy.deepcopy(tracker) for tracker in self.trackers if tracker.age > min_age]
        trajectories = [list() for _ in range(len(trackers))]

        for _ in range(future_len):
            for index, tracker in enumerate(trackers):
                trajectories[index].append(tracker.get_state())
                tracker.update_with_estimation()

        def process_trajectory(trajectory: List) -> np.array:
            return np.array([point['point_of_interest'] for point in trajectory]).astype(np.int)

        trajectories_array = [process_trajectory(trajectory) for trajectory in trajectories]
        trajectories_color = [trajectory[0]['color'] for trajectory in trajectories]

        return trajectories_array, trajectories_color

    def _update_trackers(self, detections: List[Dict]):
        matched, unmatched_detections, unmatched_trackers = \
            self._assign_detections_to_trackers(
                trackers=self.trackers, detections=detections
            )

        # Process matched detections
        if matched.size > 0:
            for tracking_index, detection_index in matched:
                detection = detections[detection_index]
                self.trackers[tracking_index].update(detection)

        # Process unmatched detections
        if len(unmatched_detections) > 0:
            for index in unmatched_detections:
                detection = detections[index]

                # assign new filter for newly discovered object
                new_tracker = KalmanTracker(
                    detection,
                    state_noise=self.state_noise, r_scale=self.r_scale, q_var=self.q_var,
                    color=(np.array(next(self.tracker_palette)) * 255).astype(np.int)
                )

                self.trackers.append(new_tracker)

        # Process unmatched tracks
        if len(unmatched_trackers) > 0:
            for tracking_index in unmatched_trackers:
                self.trackers[tracking_index].update_with_estimation()

    def _assign_detections_to_trackers(self, trackers, detections) -> \
            Tuple[np.array, np.array, np.array]:

        # constructing distance matrix
        IOU_mat = np.zeros((len(trackers), len(detections)), dtype=np.float32)
        for tracker_index, tracker in enumerate(trackers):
            for detection_index, detection in enumerate(detections):
                IOU_mat[tracker_index, detection_index] = self._get_iou(tracker, detection)

        # assigning detections to trackers
        row_indices, column_indices = linear_sum_assignment(-IOU_mat)

        unmatched_trackers = [index for index in range(len(trackers)) if index not in row_indices]
        unmatched_detections = [index for index in range(len(detections)) if index not in column_indices]

        # filtering matches with low threshold
        matches = list()
        for row, column in zip(row_indices, column_indices):
            if IOU_mat[row, column] < self.iou_threshold:
                unmatched_trackers.append(row)
                unmatched_detections.append(column)
            else:
                matches.append(np.array([row, column]).reshape(1, 2))

        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)

        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

    def _get_iou(self, tracker: KalmanTracker, detection: Dict) -> float:
        tracker_center = tracker.get_estimation()
        width, height = tracker.get_box_size()

        tracker_box = [
            tracker_center[0] - width // 2, tracker_center[1] - height // 2,
            tracker_center[0] + width // 2, tracker_center[1] + height // 2
        ]

        box = detection['box']

        detection_box = [box[0][0], box[0][1], box[1][0], box[1][1]]

        iou = self._intersection_over_union(tracker_box, detection_box)

        return iou

    @staticmethod
    def _intersection_over_union(box_a: List, box_b: List) -> float:
        # determine the (x, y)-coordinates of the intersection rectangle
        x_a = max(box_a[0], box_b[0])
        y_a = max(box_a[1], box_b[1])
        x_b = min(box_a[2], box_b[2])
        y_b = min(box_a[3], box_b[3])

        # compute the area of intersection rectangle
        intersection_area = abs(max((x_b - x_a, 0)) * max((y_b - y_a), 0))

        if intersection_area == 0:
            return 0

        # compute the area of both the prediction and ground-truth
        # rectangles
        boxaarea = abs((box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
        boxbarea = abs((box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = intersection_area / float(boxaarea + boxbarea - intersection_area)

        # return the intersection over union value
        return iou


if __name__ == "__main__":
    # uber_tracker = IOUModelTracker(
    uber_tracker = KalmanTracker(
        {'center': [100, 100], 'box': [[50, 50], [150, 150]], 'nose': [75, 75]}
    )

    uber_tracker_copy = copy.deepcopy(uber_tracker)

    print(uber_tracker.get_estimation())

    for measurement in (
        {'center': [100, 98.0], 'box': [[50, 50], [150, 150]], 'nose': [75, 75]},
        {'center': [105.0, 120.0], 'box': [[50, 50], [150, 150]], 'nose': [75, 75]},
        {'center': [107.0, 144.3], 'box': [[50, 50], [150, 150]], 'nose': [75, 75]},
        {'center': [108.0, 161.0], 'box': [[50, 50], [150, 150]], 'nose': [75, 75]},
        {'center': [108.0, 190.0], 'box': [[50, 50], [150, 150]], 'nose': [75, 75]},
        {'center': [108.0, 190.0], 'box': [[50, 50], [150, 150]], 'nose': [75, 75]},
        {'center': [108.0, 190.0], 'box': [[50, 50], [150, 150]], 'nose': [75, 75]}
    ):
        uber_tracker.update(detection=measurement)
        print(uber_tracker.get_state())

    print('\n\n\n')

    for i in range(10):
        uber_tracker.update_with_estimation()
        print(uber_tracker.get_state())

    print()

    for i in range(10):
        uber_tracker_copy.update_with_estimation()
        print(uber_tracker_copy.get_state())
