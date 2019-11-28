from collections import deque
from scipy.optimize import linear_sum_assignment
import numpy as np
from model import ModelWrapper
from filterpy.kalman import KalmanFilter, ExtendedKalmanFilter, EnsembleKalmanFilter
from filterpy.common import Q_discrete_white_noise
import seaborn as sns
import itertools
from sklearn.linear_model import LinearRegression, Lasso


class KalmanWrapper:
    def __init__(self, coors, state_noise=1.0, r_scale=1.0, q_var=1.0):
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

    def get_coors(self):
        return self.filter.x[0], self.filter.x[2]


class IOUKalmanTracker:
    def __init__(self, detection, state_noise=1.0, r_scale=1.0, q_var=1.0, color=(255, 255, 255)):
        self.color = color
        self.center_filter = KalmanWrapper(
            detection['center'],
            state_noise=state_noise, r_scale=r_scale, q_var=q_var
        )

        self.nose_filter = KalmanWrapper(
            detection['nose'],
            state_noise=state_noise, r_scale=r_scale, q_var=q_var
        )
        self.box_size = self.box_to_size(detection['box'])

        self.hits = 0
        self.no_losses = 0

    def box_to_size(self, box):
        return box[1][0] - box[0][0], box[1][1] - box[0][1]

    def update(self, detection):
        self.center_filter.predict()
        self.center_filter.update(detection['center'])
        self.nose_filter.predict()
        self.nose_filter.update(detection['nose'])
        self.box_size = self.box_to_size(detection['box'])

        # print('P', self.nose_filter.filter.P)

    def update_with_estimation(self):
        self.center_filter.predict()
        self.nose_filter.predict()

    def get_state(self):
        return {
            'center': self.center_filter.get_coors(),
            'box_size': self.get_box_size(),
            'nose': self.nose_filter.get_coors(),
            'color': self.color
        }

    def get_estimation(self):
        return self.center_filter.get_coors()

    def get_box_size(self):
        return self.box_size


class IOUModelTracker:
    def __init__(self, detection, color=(255, 0, 0)):
        self.center_queue = deque()
        self.nose_queue = deque()
        self.color = color
        self.center_queue.append(detection['center'])
        self.nose_queue.append(detection['nose'])
        self.box_size = self.box_to_size(detection['box'])

        file_path_pattern = '/home/marcus/data/sber/lr_len_{}.pkl'
        # file_path_pattern = 'lr_len_{}.pkl'
        self.lengths = (2, 4, 8, 16, 32, 64)
        self.models = [ModelWrapper(file_path_pattern, length) for length in self.lengths]
        self.hits = 0
        self.no_losses = 0

    def box_to_size(self, box):
        return box[1][0] - box[0][0], box[1][1] - box[0][1]

    def _predict_next_detection(self, queue):
        if len(queue) == 1:
            return queue[-1]

        current_model = self._choose_right_model()
        return current_model.predict(queue)

    # @staticmethod
    # def _predict_next_detection(queue):
    #     if len(queue) == 1:
    #         return queue[-1]
    #
    #     def smooth_data(traj):
    #         x_train, y_train = traj[:, 0].reshape((-1, 1)), traj[:, 1]
    #         huber = LinearRegression()
    #         huber.fit(x_train, y_train)
    #         prediction = huber.predict(x_train).reshape((-1, 1))
    #         return np.hstack((x_train, prediction))
    #
    #     data = np.array(queue)
    #     data = smooth_data(data)
    #     velocity = (data[-1, :] - data[0, :]) / data.shape[0]
    #
    #     prediction = data[-1, :] + velocity
    #
    #     return prediction.flatten()

    def _choose_right_model(self):
        right_index = 0
        for index, model_length in enumerate(self.lengths):
            if model_length <= len(self.center_queue):
                right_index = index
            else:
                break
        return self.models[right_index]

    def update(self, detection):
        self.center_queue.append(detection['center'])
        self.nose_queue.append(detection['nose'])
        self.box_size = self.box_to_size(detection['box'])

    def update_with_estimation(self):
        next_center = self._predict_next_detection(self.center_queue)
        next_nose = self._predict_next_detection(self.nose_queue)
        self.center_queue.append(next_center)
        self.nose_queue.append(next_nose)

    def get_state(self):
        return {
            'center': self.center_queue[-1],
            'nose': self.nose_queue[-1],
            'color': self.color
        }

    def get_estimation(self):
        return self._predict_next_detection(self.center_queue)

    def get_box_size(self):
        return self.box_size


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


class IOUModelTracking:
    def __init__(self, kalman=False, state_noise=40.0, r_scale=10.0, q_var=1.0, iou_threshold=0.3, max_age=4, min_hits=1):
        self.kalman = kalman
        self.state_noise = state_noise
        self.r_scale = r_scale
        self.q_var = q_var

        self.iou_threshold = iou_threshold
        self.frame_count = 0  # frame counter
        self.max_age = max_age  # no.of consecutive unmatched detection before a track is deleted
        self.min_hits = min_hits  # no. of consecutive matches needed to establish a track
        self.tracker_list = list()  # list for trackers
        self.tracker_palette = itertools.cycle(sns.color_palette())

    def get_IOU(self, tracker, detection):
        tracker_center = tracker.get_estimation()
        width, height = tracker.get_box_size()

        tracker_box = [
            tracker_center[0] - width // 2, tracker_center[1] - height // 2,
            tracker_center[0] + width // 2, tracker_center[1] + height // 2
        ]

        box = detection['box']

        detection_box = [box[0][0], box[0][1], box[1][0], box[1][1]]

        iou = bb_intersection_over_union(tracker_box, detection_box)

        return iou

    def assign_detections_to_trackers(self, trackers, detections, iou_threshold=20):
        """
        From current list of trackers and new detections, output matched detections,
        unmatchted trackers, unmatched detections.
        """
        IOU_mat = np.zeros((len(trackers), len(detections)), dtype=np.float32)

        for tracker_index, tracker in enumerate(trackers):
            for detection_index, detection in enumerate(detections):
                IOU_mat[tracker_index, detection_index] = self.get_IOU(tracker, detection)

        row_ind, col_ind = linear_sum_assignment(-IOU_mat)

        unmatched_trackers = list()
        unmatched_detections = list()

        for tracking_index, tracker in enumerate(trackers):
            if tracking_index not in row_ind:
                unmatched_trackers.append(tracking_index)

        for detection_index, detection in enumerate(detections):
            if detection_index not in col_ind:
                unmatched_detections.append(detection_index)

        matches = list()

        for m in zip(row_ind, col_ind):
            if IOU_mat[m[0], m[1]] < iou_threshold:
                unmatched_trackers.append(m[0])
                unmatched_detections.append(m[1])
            else:
                matches.append(np.array(m).reshape(1, 2))

        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)

        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

    def track(self, detections):
        """
        Pipeline function for detection and tracking
        """
        self.frame_count += 1

        if len(self.tracker_list) == 0 and len(detections) == 0:
            return list()

        trackers = list()

        if len(self.tracker_list) > 0:
            for tracker in self.tracker_list:
                trackers.append(tracker)

        matched, unmatched_detections, unmatched_trackings = \
            self.assign_detections_to_trackers(
                trackers=trackers, detections=detections, iou_threshold=self.iou_threshold
            )

        # Deal with matched detections
        if matched.size > 0:
            for tracking_index, detection_index in matched:
                detection = detections[detection_index]
                tmp_tracker = self.tracker_list[tracking_index]
                tmp_tracker.update(detection)
                tmp_tracker.hits += 1
                tmp_tracker.no_losses = 0

        # Deal with unmatched detections
        if len(unmatched_detections) > 0:
            for index in unmatched_detections:
                detection = detections[index]

                if self.kalman:
                    # raise ValueError
                    new_tracker = IOUKalmanTracker(
                        detection,
                        state_noise=self.state_noise, r_scale=self.r_scale, q_var=self.q_var,
                        color=(np.array(next(self.tracker_palette)) * 255).astype(np.int)
                    )
                else:
                    new_tracker = IOUModelTracker(
                        detection,
                        color=(np.array(next(self.tracker_palette)) * 255).astype(np.int)
                    )

                self.tracker_list.append(new_tracker)

        # Deal with unmatched tracks
        if len(unmatched_trackings) > 0:
            for tracking_index in unmatched_trackings:
                tmp_tracker = self.tracker_list[tracking_index]
                tmp_tracker.no_losses += 1
                tmp_tracker.update_with_estimation()

        # The list of tracks to be annotated
        good_detections = list()

        for tracker in self.tracker_list:
            if tracker.hits >= self.min_hits and tracker.no_losses <= self.max_age:
                good_detections.append(tracker.get_state())

        self.tracker_list = [x for x in self.tracker_list if x.no_losses <= self.max_age]

        return good_detections


if __name__ == "__main__":
    # uber_tracker = IOUModelTracker(
    uber_tracker = IOUKalmanTracker(
        {'center': [100, 100], 'box': [[50, 50], [150, 150]], 'nose': [75, 75]}
    )

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
        print()

    print('\n\n\n')

    for i in range(10):
        uber_tracker.update_with_estimation()
        print(uber_tracker.get_state())
        print()
