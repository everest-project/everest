#! /bin/bash
mkdir cached_gt
mkdir videos
mkdir weights
gdown https://drive.google.com/uc?id=1uO3HbeD313Aaff5L_2OLhau1_VWKW5td -O cached_gt/traffic_footage_number_of_cars.npy
gdown https://drive.google.com/uc?id=1YXJfZv8KXbSdb_gSMJhz-fj8RuPer5Hs -O videos/traffic_footage.mp4
wget -c "https://pjreddie.com/media/files/yolov3.weights" --header "Referer: pjreddie.com" -O weights/yolov3.weights
wget -c "https://pjreddie.com/media/files/yolov3-tiny.weights" --header "Referer: pjreddie.com" -O weights/yolov3-tiny.weights
