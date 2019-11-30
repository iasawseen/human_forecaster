import cv2
from tracking import Tracking
from tqdm import tqdm
import os
import numpy as np
from model import PoseDetector, HeadDetector
from utils import save_video, video_iter, set_seed
from configs.config import get_cfg

# VIDEO_FILE_NAME = '/home/marcus/data/sber/TUD-Stadtmitte.mp4'
# VIDEO_FILE_NAME = '/home/marcus/data/sber/MOT17-09-SDP.mp4'
VIDEO_FILE_NAME = 'MOT17-09-SDP.mp4'
# VIDEO_FILE_NAME = '/home/marcus/data/sber/MOT17-08-SDP.mp4'
# VIDEO_FILE_NAME = '/home/marcus/data/sber/MOT16-03.mp4'
# VIDEO_FILE_NAME = '/home/marcus/data/sber/people_walking_russia.mp4'
# VIDEO_FILE_NAME = 'people_walking_russia.mp4'
# VIDEO_FILE_NAME = '/home/marcus/data/sber/P1033656.mp4'
# VIDEO_FILE_NAME = '/home/marcus/data/sber/P1033756.mp4'
# VIDEO_FILE_NAME = '/home/marcus/data/sber/forecaster/P1044014.mp4'
# VIDEO_FILE_NAME = '/home/marcus/data/sber/forecaster/P1033669.mp4'
# VIDEO_FILE_NAME = '/home/marcus/data/sber/forecaster/P1033788.mp4'


# class VideoProcessor

def draw_tracked_detections(frame, tracked_detections):
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
            cfg.OUTPUT_VIDEO.BLOB_SIZE, color.tolist(), -1
        )

        if cfg.OUTPUT_VIDEO.DRAW_BOX:
            cv2.rectangle(
                frame,
                (x_min, y_min), (x_max, y_max),
                color.tolist(), cfg.OUTPUT_VIDEO.LINE_WIDTH)


def draw_trajectories(cfg, frame, trajectories, colors, frames=None):
    for time_index in range(1, cfg.OUTPUT_VIDEO.CYCLE_LEN):
        if frames is not None:
            frame = np.array(frame)

        for traj, color in zip(trajectories, colors):
            x_start, y_start = traj[time_index - 1]
            x_end, y_end = traj[time_index]

            cv2.line(
                frame, (x_start, y_start), (x_end, y_end),
                color.tolist(), cfg.OUTPUT_VIDEO.LINE_WIDTH
            )

        if frames is not None:
            frames.append(frame)

    return frame


def demo(cfg):

    tracking = Tracking(
        cfg,
        state_noise=cfg.TRACKING.STATE_NOISE,
        r_scale=cfg.TRACKING.R_SCALE,
        q_var=cfg.TRACKING.Q_VAR,
        iou_threshold=cfg.TRACKING.IOU_THRESHOLD,
        max_age=cfg.TRACKING.MAX_AGE,
        min_hits=cfg.TRACKING.MIN_HITS
    )

    tracked_frames = list()
    draw_tracks = False
    trajectories, colors = None, None

    for i, next_frame in tqdm(enumerate(video_iter(VIDEO_FILE_NAME))):
        next_frame_to_visual = np.array(next_frame)

        if draw_tracks and i % cfg.OUTPUT_VIDEO.CYCLE_LEN != 0:
            draw_trajectories(cfg, next_frame_to_visual, trajectories=trajectories, colors=colors)

        if i != 0 and i % cfg.OUTPUT_VIDEO.CYCLE_LEN == 0:
            draw_tracks = True
            trajectories, colors = tracking.predict_trajectories(
                cfg.OUTPUT_VIDEO.CYCLE_LEN, min_age=cfg.OUTPUT_VIDEO.MIN_AGE_FOR_TRAJECTORY
            )

            next_frame_to_visual = draw_trajectories(
                cfg, next_frame_to_visual,
                trajectories=trajectories, colors=colors, frames=tracked_frames
            )

        tracked_detections = tracking.track(next_frame)
        draw_tracked_detections(next_frame_to_visual, tracked_detections)

        tracked_frames.append(next_frame_to_visual)

    save_video(tracked_frames, fps=cfg.OUTPUT_VIDEO.FPS, file_name='tracked_heads.mp4')

    os.system('ffmpeg -i tracked_heads.mp4 -vcodec libx264 tracked_heads_2.mp4')
    os.system('mv -f tracked_heads_2.mp4 tracked_heads.mp4')


if __name__ == '__main__':
    cfg = get_cfg()
    set_seed(cfg.MAIN.SEED)
    demo(cfg)
