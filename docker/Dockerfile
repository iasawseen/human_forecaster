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

RUN pip install \
    opencv-python \
    scipy \
    filterpy \
    seaborn \
    sklearn \
    yacs \
    attrdict

RUN conda install cython -y && conda clean --all

WORKDIR /tmp/for_mmdetection
RUN git clone -b head_detection https://github.com/iasawseen/mmdetection.git
WORKDIR /tmp/for_mmdetection/mmdetection
RUN pip install mmcv
RUN pip install -v -e .

WORKDIR /human_forecaster