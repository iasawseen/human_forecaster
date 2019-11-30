FROM pytorch/pytorch:1.2-cuda10.0-cudnn7-devel

RUN apt-get update && apt-get install -y \
    ffmpeg \
    build-essential \
    libsm6 \
    libxext6 \
    libfontconfig1 \
    libxrender1 \
    libswscale-dev \
    libtbb2 \
    libtbb-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libjasper-dev \
    libavformat-dev \
    libpq-dev \
    libturbojpeg \
    software-properties-common \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN pip install opencv-python
RUN pip install scipy
RUN pip install filterpy
RUN pip install seaborn
RUN pip install sklearn
RUN pip install yacs
RUN pip install attrdict

WORKDIR /pose_forecaster