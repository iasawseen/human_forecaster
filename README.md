# Simple Human Tracking and Trajectory Prediction

This repository presents simple approach for human tracking and trajectory prediction based on human keypoint detection and Kalman filter.

# Hardware 
Cuda compatible video card: this repository is tested on RTX 2080Ti

# Software 
Please check the `docker/Dockerfile`.  


# Getting started 
The sole task of this project is to demonstrate how human detection with linear Kalman filtering can be combined into tracking and trajectory prediction.
Here are some examples:

![Example0](images/example0.gif)
![Example1](images/example1.gif)

Every n seconds future trajectories are predicted and drawn on current frame. 

To reproduce these results go to next sections.

# Install Docker
Thirst of all install Docker:
```
sudo apt install docker.io
```
After that install [nvidia-docker v2.0](<https://github.com/NVIDIA/nvidia-docker/wiki/Installation-(version-2.0)>):
```
docker volume ls -q -f driver=nvidia-docker | xargs -r -I{} -n1 docker ps -q -a -f volume={} | xargs -r docker rm -f
sudo apt-get purge nvidia-docker
``` 

# Build Docker Image
```bash
cd docker 
sudo docker build . -t sawseen/pytorch_cv:pose_forecaster
cd ..
```

# Run container

To start working in container run the following commands: 

```bash
make run 
make exec 
```

After you have done working:
```bash
make stop 
```

# Settings 

The settings are located in `configs/config.yaml` file. 

```yml 
MAIN:
  SEED: 42
  HEAD_DETECTION: True

DETECTING:
  HEAD_CONFIG_FILE_PATH: './configs/head_cascade_rcnn_dconv_c3-c5_r50_fpn_1x.py'
  HEAD_CHECKPOINT_FILE_PATH: './models/cascade_epoch_11.pth'
  SCORE_THRESHOLD: 0.75
  NMS_IOU_THRESHOLD: 0.2

TRACKING:
  STATE_NOISE: 1000.0
  R_SCALE: 5.0
  Q_VAR: 100.0
  IOU_THRESHOLD: 0.1
  MAX_AGE: 6
  MIN_HITS: 2

OUTPUT_VIDEO:
  CYCLE_LEN: 2
  BLOB_SIZE: 4
  LINE_WIDTH: 8
  FPS: 16
  MIN_AGE_FOR_TRAJECTORY: 12
  DRAW_BOX: True
```

1. `MAIN` section consists of `SEED` for reproducibility and `HEAD_DETECTION` whether to use model trained on human heads.

2. `DETECTING` describe how detections are made. `HEAD_CONFIG_FILE_PATH` and `HEAD_CHECKPOINT_FILE_PATH` are required for head detection model. 
`SCORE_THRESHOLD` is a detection threshold, while `NMS_IOU_THRESHOLD` is an intersection over union threshold for non-maximum suppression.
 
3. `TRACKING` is responsible for tracking parameters. `STATE_NOISE`, `R_SCALE`, `Q_VAR` are Kalman filter parameters. 
`IOU_THRESHOLD` is an intersection over union threshold for assuming whether detection and predicted state of current tracker are matched. 
`MAX_AGE` is a parameter that tells how long a tracker will live without any matched detections.
`MIN_HITS` is a parameter that tells minimum number of frames for a tracker to be drawn.

4. `OUTPUT_VIDEO` specifies parameters of the output video. `CYCLE_LEN` is period of predicted trajectory. 
`BLOB_SIZE` and `LINE_WIDTH` are sizes of blob around human anchor point and trajectory line.
`FPS` is desired frames per second for output video. `MIN_AGE_FOR_TRAJECTORY` is a minimum age for tracker in frames that trajectory will be drawn.
`DRAW_BOX` states whether to draw bounding boxes around objects.

# Demo

```bash
python demo.py input_videl.mp4 --output-video output_video.mp4 --prediction-length 2.0 --head-detection --draw-boxes
``` 
Where: 
* `output-video`: output video file name,
* `prediction-length`: length of trajectories in seconds (overrides config parameter), 
* `head-detection`: flag whether to detect heads (overrides config parameter),
* `draw-boxes`: flag whether to draw bounding boxes around objects. 