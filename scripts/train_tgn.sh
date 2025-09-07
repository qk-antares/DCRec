#!/bin/bash
export PYTHONPATH=$PYTHONPATH:~/workplace/python/DCRec/
nohup python main.py \
  --model tgn \
  --data ml-100k \
  --use_memory \
  --prefix tgn \
  --n_neg 10 \
  --n_test_neg 100 \
  --memory_dim 173 \
  > train_tgn.log 2>&1 &