#!/bin/bash
export PYTHONPATH=$PYTHONPATH:~/workplace/python/DCRec/
cd ~/workplace/python/DCRec/
nohup python main.py \
  --data ml-100k \
  --use_memory \
  --prefix tgn-test \
  --n_neg 1 \
  --n_epoch 3 \
  --n_skip_val 0 \
  --memory_dim 173 \
  --n_test_neg 100 \
  > test-tgn-ml-100k-neg1_100.log 2>&1 &

