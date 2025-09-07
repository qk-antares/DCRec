#!/bin/bash
export PYTHONPATH=$PYTHONPATH:~/workplace/python/DCRec/
cd ~/workplace/python/DCRec/
nohup python main.py \
  --model dcrec \
  --data ml-100k \
  --n_test_neg 100 \
  --memory_dim 173 \
  --use_memory \
  --prefix dcrec \
  --n_neg 100 \
  --noise_pruning_ratio 0.05 \
  > train_dcrec.log 2>&1 &