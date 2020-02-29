# Everest

This is the official project page for Everest project.

## Requirements

You will need the following installed:

- python>=3.6.8
- CUDA>=9.0
- CUDNN>=7.6.0
- tensorflow-gpu>=1.4.0
- Opencv 3.2 with FFmpeg bindings
- g++ 4.8.5 or later 

Your machine will need at least:

- A GPU (this has only been tested with NVIDIA GeForce GTX 1080 Ti)
- 64+ GB of memory
- 500+ GB of disk space 
- AVX2 capabilities

## Guides on Installing the Requirements 

- python 3.6 - For Linux, use your package manager.
- CUDA, CUDNN, tensorflow-gpu
-- 
```sh
git clone https://github.com/xchani/everest.git
cd everest
pip3 install -r requirements.txt --user
```

## Generating Queries

```sh
bash run.sh ${config}
```

