#! /bin/bash

data_config=$1

python split_dataset.py --data_config $data_config && \
python train_yolomdn.py --data_config $data_config && \
python inference.py --data_config $data_config && \
python gen_cdf.py --data_config $data_config && \
python topk.py --data_config $data_config 

