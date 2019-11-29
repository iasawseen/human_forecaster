FROM pytorch/pytorch:1.2-cuda10.0-cudnn7-devel

RUN apt-get update && apt-get install -y \
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

COPY nccl-repo-ubuntu1604-2.4.8-ga-cuda10.0_1-1_amd64.deb /tmp/nccl-repo-ubuntu1604-2.4.8-ga-cuda10.0_1-1_amd64.deb
RUN dpkg -i /tmp/nccl-repo-ubuntu1604-2.4.8-ga-cuda10.0_1-1_amd64.deb

RUN apt-get update
RUN apt-get install libnccl2=2.4.8-1+cuda10.0 libnccl-dev=2.4.8-1+cuda10.0

RUN conda install cython -y && conda clean --all

WORKDIR /tmp/for_mmdetection
RUN git clone https://github.com/open-mmlab/mmdetection.git
WORKDIR /tmp/for_mmdetection/mmdetection
RUN pip install mmcv
RUN pip install -v -e .

RUN pip install yacs
RUN pip install attrdict

RUN apt-get install -y ffmpeg

# Compile and install ffmpeg from source
#RUN git clone https://github.com/FFmpeg/FFmpeg /root/ffmpeg && \
#    cd /root/ffmpeg && \
#    ./configure --enable-nonfree --disable-shared --extra-cflags=-I/usr/local/include && \
#    make -j8 && make install -j8

WORKDIR /pose_forecaster