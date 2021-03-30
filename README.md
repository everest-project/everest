# Everest

This is the referece implementation of Everest described in:

**[SIGMOD 2021] Top-K Deep Video Analytics: A Probabilistic Approach** [[arxiv]](https://arxiv.org/abs/2003.00773)

*Authors: Ziliang Lai, Chenxia Han, Chris Liu, Pengfei Zhang, Eric Lo, Ben Kao*


## Recommended Hardware

- CPU: Intel i9-7900X or above
- GPU: NVIDIA GTX1080Ti or above
- RAM: >= 64GB

## Setup
```sh
git clone https://github.com/everest-project/everest.git
cd everest
docker pull zllai/everest:1.2
```

## Run a sample query
Query: find top-100 frames with largest number of cars in a 5-hour traffic footage
1. Download the sample files:
```sh
./tools/download_sample.sh
```
2. Run the container:
```sh
./tools/docker.sh
```
3. Run the query:
```sh
python3 everest.py @config/traffic_footage.arg
```
