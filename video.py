import cv2
from tracking import Tracking
from tqdm import tqdm
import os
import numpy as np


class VideoProcessor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.tracked_frames = None

    def process(self, video_file_path):
        tracking = Tracking(
            self.cfg,
            state_noise=self.cfg.TRACKING.STATE_NOISE,
            r_scale=self.cfg.TRACKING.R_SCALE,
            q_var=self.cfg.TRACKING.Q_VAR,
            iou_threshold=self.cfg.TRACKING.IOU_THRESHOLD,
            max_age=self.cfg.TRACKING.MAX_AGE,
            min_hits=self.cfg.TRACKING.MIN_HITS
        )

        self.tracked_frames = list()
        draw_tracks = False
        trajectories, colors = None, None

        video_iterator = self.video_iter(video_file_path)
        fps = int(next(video_iterator))

        future_len = int(self.cfg.OUTPUT_VIDEO.CYCLE_LEN * fps)

        for i, next_frame in tqdm(enumerate(video_iterator)):
            next_frame_to_visual = np.array(next_frame)

            if draw_tracks and i % future_len != 0:
                self.draw_trajectories(
                    next_frame_to_visual,
                    trajectories=trajectories, future_len=future_len, colors=colors
                )

            if i != 0 and i % future_len == 0:
                draw_tracks = True
                trajectories, colors = tracking.predict_trajectories(
                    future_len, min_age=self.cfg.OUTPUT_VIDEO.MIN_AGE_FOR_TRAJECTORY
                )

                next_frame_to_visual = self.draw_trajectories(
                    next_frame_to_visual,
                    trajectories=trajectories, colors=colors, future_len=future_len, save_intermediate=True
                )

            tracked_detections = tracking.track(next_frame)
            self.draw_tracked_detections(next_frame_to_visual, tracked_detections)
            self.tracked_frames.append(next_frame_to_visual)

    @staticmethod
    def video_iter(file_path):
        video = cv2.VideoCapture(file_path)
        fps = video.get(cv2.CAP_PROP_FPS)

        yield fps

        success, image = video.read()

        if success:
            yield image

        while success:
            success, image = video.read()
            if success:
                yield image

    def draw_tracked_detections(self, frame, tracked_detections):
        for detection in tracked_detections:
            center = detection['nose']
            color = detection['color']
            box_center = detection['center']
            width, height = detection['box_size']

            x_min, x_max = int(box_center[0] - width // 2), int(box_center[0] + width // 2)
            y_min, y_max = int(box_center[1] - height // 2), int(box_center[1] + height // 2)

            cv2.circle(
                frame,
                (int(center[0]), int(center[1])),
                self.cfg.OUTPUT_VIDEO.BLOB_SIZE, color.tolist(), -1
            )

            if self.cfg.OUTPUT_VIDEO.DRAW_BOX:
                cv2.rectangle(
                    frame,
                    (x_min, y_min), (x_max, y_max),
                    color.tolist(), self.cfg.OUTPUT_VIDEO.LINE_WIDTH)

    def draw_trajectories(self, frame, trajectories, colors, future_len, save_intermediate=False):
        for time_index in range(1, future_len):
            if save_intermediate:
                frame = np.array(frame)

            for traj, color in zip(trajectories, colors):
                x_start, y_start = traj[time_index - 1]
                x_end, y_end = traj[time_index]

                cv2.line(
                    frame, (x_start, y_start), (x_end, y_end),
                    color.tolist(), self.cfg.OUTPUT_VIDEO.LINE_WIDTH
                )

            if save_intermediate:
                self.tracked_frames.append(frame)

        return frame

    def save_video(self, file_path):
        if len(self.tracked_frames) == 0:
            return

        size = self.tracked_frames[0].shape[1], self.tracked_frames[0].shape[0]

        out = cv2.VideoWriter(
            file_path,
            fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
            fps=self.cfg.OUTPUT_VIDEO.FPS,
            frameSize=size
        )

        for frame in self.tracked_frames:
            out.write(frame)

        out.release()

        os.system(f'ffmpeg -i {file_path} -vcodec libx264 tmp.mp4')
        os.system(f'mv -f tmp.mp4 {file_path}')
