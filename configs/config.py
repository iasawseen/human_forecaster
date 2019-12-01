from yacs.config import CfgNode as CN
from attrdict import AttrDict

_C = CN()

_C.MAIN = CN()
_C.MAIN.SEED = 42
_C.MAIN.HEAD_DETECTION = False

_C.DETECTING = CN()
_C.DETECTING.HEAD_CONFIG_FILE_PATH = './configs/head_cascade_rcnn_dconv_c3-c5_r50_fpn_1x.py'
_C.DETECTING.HEAD_CHECKPOINT_FILE_PATH = './models/cascade_epoch_11.pth'
_C.DETECTING.SCORE_THRESHOLD = 0.75
_C.DETECTING.NMS_IOU_THRESHOLD = 0.2

_C.TRACKING = CN()

_C.TRACKING.STATE_NOISE = 1000.0
_C.TRACKING.R_SCALE = 5.0
_C.TRACKING.Q_VAR = 100.0
_C.TRACKING.IOU_THRESHOLD = 0.1
_C.TRACKING.MAX_AGE = 6
_C.TRACKING.MIN_HITS = 2

_C.OUTPUT_VIDEO = CN()

_C.OUTPUT_VIDEO.CYCLE_LEN = 32
_C.OUTPUT_VIDEO.BLOB_SIZE = 8
_C.OUTPUT_VIDEO.LINE_WIDTH = 16
_C.OUTPUT_VIDEO.FPS = 8
_C.OUTPUT_VIDEO.MIN_AGE_FOR_TRAJECTORY = 1.0
_C.OUTPUT_VIDEO.DRAW_BOX = False
_C.OUTPUT_VIDEO.COMPRESS = False


_C.CONFIG_TO_MERGE = 'configs/config.yaml'


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values"""
    return _C.clone()


def get_cfg():
    cfg = get_cfg_defaults()
    cfg.merge_from_file(cfg.CONFIG_TO_MERGE)

    return cfg


if __name__ == '__main__':
    cfg = get_cfg()
