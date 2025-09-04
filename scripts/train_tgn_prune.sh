#!/bin/bash
export PYTHONPATH=$PYTHONPATH:~/workplace/python/DCRec/
nohup python main.py \
  --use_memory \
  --prefix tgn-prune \
  --n_neg 10 \
  --noise_pruning_ratio 0.1 \
  > train_tgn_prune.log 2>&1 &