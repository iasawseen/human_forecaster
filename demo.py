from utils import set_seed
from configs.config import get_cfg
from video import VideoProcessor
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_video', type=str, help='file_path for input video')
    parser.add_argument('-o', '--output-video', type=str, default='processed.mp4', help='file_path for output video')
    parser.add_argument('--head-detection', help='detect only heads', action='store_true')
    parser.add_argument(
        '-l', '--prediction-length', type=float, default=1.0,
        help='number of seconds to predict trajectory in the future'
    )

    return parser.parse_args()


if __name__ == '__main__':
    cfg = get_cfg()
    set_seed(cfg.MAIN.SEED)
    args = parse_args()

    cfg.MAIN.HEAD_DETECTION = args.head_detection
    cfg.OUTPUT_VIDEO.CYCLE_LEN = args.prediction_length

    processor = VideoProcessor(cfg)
    processor.process(args.input_video)
    processor.save_video(args.output_video)
