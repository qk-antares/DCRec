#!/bin/bash
export PYTHONPATH=$PYTHONPATH:~/workplace/python/DCRec/
cd ~/workplace/python/DCRec/
nohup python main.py \
  --use_memory \
  --prefix test-tgn-l2-0.0001 \
  --n_neg 10 \
  --train_ratio 0.05 \
  --valid_ratio 0.01 \
  --n_epoch 3 \
  --n_skip_val 0 \
  --l2_regularization 0.0001 \
  > test_tgn_l2_0.0001.log 2>&1 &
