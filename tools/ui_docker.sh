#! /bin/bash
#docker run -it --rm -v /home/zllai/raw_video:/mnt/raw_video -v /home/zllai/video-analytic/resized_video:/mnt/resized_video -v /home/hnchen/everest:/home/zllai/video-analytic --name everest_container --gpus all -p 3000:3000 everest_env bash
docker run -it --rm -v /home/zllai/raw_video:/mnt/raw_video -v /home/hnchen/everest/videos:/mnt/resized_video -v /home/hnchen/everest:/home/zllai/video-analytic --name everest_container --gpus all -p 3000:3000 everest_env bash
