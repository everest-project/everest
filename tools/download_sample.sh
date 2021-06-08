#! /bin/bash
mkdir cached_gt
mkdir videos
mkdir weights
gdown https://drive.google.com/uc?id=1uO3HbeD313Aaff5L_2OLhau1_VWKW5td -O cached_gt/traffic_footage_number_of_cars.npy
gdown https://drive.google.com/uc?id=1YXJfZv8KXbSdb_gSMJhz-fj8RuPer5Hs -O videos/traffic_footage.mp4
wget -c "https://pjreddie.com/media/files/yolov3.weights" --header "Referer: pjreddie.com" -O weights/yolov3.weights
wget -c "https://pjreddie.com/media/files/yolov3-tiny.weights" --header "Referer: pjreddie.com" -O weights/yolov3-tiny.weights

# yolov3-spp5-custom_best.weights for exciting_moment: https://drive.google.com/file/d/1lZUPYQLlCzvxgxpj3GUSXjtBtpdsSJxx/view?usp=sharing
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1lZUPYQLlCzvxgxpj3GUSXjtBtpdsSJxx' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1lZUPYQLlCzvxgxpj3GUSXjtBtpdsSJxx" -O weights/yolov3-spp5-custom_best.weights && rm -rf /tmp/cookies.txt

# 1GB test video for exciting_moment: https://drive.google.com/file/d/1p-FeyS6ziojuPJ8zqidcGxN1JoxapGg-/view?usp=sharing
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1p-FeyS6ziojuPJ8zqidcGxN1JoxapGg-' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1p-FeyS6ziojuPJ8zqidcGxN1JoxapGg-" -O videos/test07.mp4 && rm -rf /tmp/cookies.txt


cd weights
for MODEL in 'hybrid_finetuned_fc6+' 'hybrid_finetuned_all' 'vgg19_finetuned_fc6+' 'vgg19_finetuned_all'; do
  if [ ! -f "${MODEL}.pth" ]; then
      echo "Downloading: ${MODEL}.pth"
      wget https://github.com/fabiocarrara/visual-sentiment-analysis/releases/download/torch-models/${MODEL}.pth
  else
      echo "Skipping: ${MODEL}.pth already downloaded"
  fi
done
cd ..

