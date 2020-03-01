# Everest

This is the official project page for Everest project.

## Requirements

You will need the following installed:

- Python>=3.7.4
- CUDA>=10.0
- CUDNN>=7.6.0
- Opencv 3.2 with FFmpeg bindings
- g++ 4.8.5 or later 

Your machine will need at least:

- A GPU (this has only been tested with NVIDIA GeForce GTX 1080 Ti)
- 64+ GB of memory
- 500+ GB of disk space 

## Guides on Installing the Requirements 

- python 3.7.4 - For Linux, we recommend that the users use [anconda](https://www.anaconda.com/).
- CUDA, CUDNN

    You can refer to [here](https://github.com/BVLC/caffe/wiki/OpenCV-3.2-Installation-Guide-on-Ubuntu-16.04) in order to install the OpenCV 3.2 with FFmpeg bindings.


## Setting up the Top-K Query Engine

To set up the query engine, do the following.

```sh
git clone https://github.com/everest-project/everest.git
cd everest
pip3 install -r requirements.txt --user
```

## Download Pretrained Weights

```sh
mkdir weights
cd weights
wget -c https://pjreddie.com/media/files/yolov3.weights
wget -c https://pjreddie.com/media/files/yolov3-tiny.weights
```

## Generating Queries

Once you have query engine set up, the ```run.sh``` is the script to run a top-k query example. 

```sh
bash run.sh ${config}
```
