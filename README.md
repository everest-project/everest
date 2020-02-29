# Everest

This is the official project page for Everest project.

## Requirements

You will need the following installed:

- python>=3.7.4
- CUDA>=10.0
- CUDNN>=7.6.0
- tensorflow-gpu>=1.14.0
- Opencv 3.2 with FFmpeg bindings
- g++ 4.8.5 or later 

Your machine will need at least:

- A GPU (this has only been tested with NVIDIA GeForce GTX 1080 Ti)
- 64+ GB of memory
- 500+ GB of disk space 
- AVX2 capabilities

## Guides on Installing the Requirements 

- python 3.7.4 - For Linux, we recommend that the users use [anconda](https://www.anaconda.com/).
- CUDA, CUDNN, tensorflow-gpu

    [TensorFlow 1.14.0](https://github.com/tensorflow/tensorflow) with CUDA 10.0 and CUDNN 7.6.0 -- WE ONLY TEST OUR CODE WITH THIS COMBINATION. 
    
    Note: Having both TensorFlow-gpu 1.14.0 and more recent versions installed is complicated. This project requires cuDNN 7.6.0 and more recent versions of TensorFlow may break with the installed. Therefore, it is recommended that users uninstall more recent versions of TensorFlow and delete other versions of cuDNN.
    
    You can refer to [here](https://github.com/BVLC/caffe/wiki/OpenCV-3.2-Installation-Guide-on-Ubuntu-16.04) in order to install the OpenCV 3.2 with FFmpeg bindings.
    

## Setting up the Top-K Query Engine

To set up the query engine, do the following.

```sh
git clone https://github.com/xchani/everest.git
cd everest
pip3 install -r requirements.txt --user
```

## Generating Queries

Once you have query engine set up, the ```run.sh``` is the script to run a top-k example. 

```sh
bash run.sh ${config}
```
Have fun!

