FROM nvidia/cuda:10.2-devel

# nvidia env
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES video,compute,utility

RUN apt-get update
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:jonathonf/ffmpeg-4
RUN apt-get install -y build-essential make cmake
RUN apt-get install -y ffmpeg libavcodec-dev libavfilter-dev libavformat-dev libavutil-dev
RUN apt-get install -y git vim curl unzip python3-pip
RUN add-apt-repository ppa:deadsnakes/ppa -y
RUN apt-get install python3.8 python3.8-dev python3.8-distutils python3-pip -y
RUN python3.8 -m pip install --upgrade pip setuptools wheel
RUN yes | pip --no-cache-dir install torch numpy Pillow tqdm lmdb torchvision blist scipy flask
RUN rm /usr/bin/python3 && ln -s /usr/bin/python3.8 /usr/bin/python3

# nvcodec
RUN curl -fsSL https://developer.download.nvidia.com/compute/redist/VideoCodec/v8.2/NvCodec.zip -O && \
    unzip -j NvCodec.zip \
          NvCodec/Lib/linux/stubs/x86_64/libnvcuvid.so \
          NvCodec/Lib/linux/stubs/x86_64/libnvidia-encode.so \
          -d /usr/local/cuda/lib64/stubs && \
    rm NvCodec.zip && \
    rm -rf NvCodec && \
    ln -s /usr/local/cuda/lib64/stubs/libnvcuvid.so /usr/lib/libnvcuvid.so

# decord
RUN git clone --depth 1 --branch v0.3.9 --recursive https://github.com/dmlc/decord /home/decord
WORKDIR /home/decord

RUN mkdir build && cd build && cmake .. -DUSE_CUDA=/usr/local/cuda -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc && make && \
    cd ../python && python3 setup.py install

RUN yes | pip --no-cache-dir install matplotlib flask
RUN curl -fsSL https://deb.nodesource.com/setup_10.x | bash -
RUN apt-get install -y nodejs

RUN yes | pip --no-cache-dir install opencv-python
RUN yes | pip --no-cache-dir install gdown
RUN apt-get install -y wget
 
WORKDIR /mnt/everest


