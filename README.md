# Everest

This is the official project page for Everest project.

## Requirements

You will need the following installed:

- python>=3.6
- CUDA>=9.0
- CUDNN>=
- Pytorch-gpu>=
- Opencv 3.2 with FFmpeg bindings
- g++ >= 

Your machine will need at least:

- A GPU (this has only been tested with NVIDIA GeForce GTX 1080 Ti)
- 64+ GB of memory
- 500+ GB of disk space 
- AVX2 capabilities

## Installation

```sh
git clone https://github.com/xchani/everest.git
cd everest
pip3 install -r requirements.txt --user
```

## Generating Queries

```sh
bash run.sh ${config}
```

