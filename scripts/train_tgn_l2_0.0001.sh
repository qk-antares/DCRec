#!/bin/bash
export PYTHONPATH=$PYTHONPATH:~/workplace/python/DCRec/
nohup python main.py \
  --use_memory \
  --prefix tgn-l2-0.0001 \
  --n_neg 10 \
  --l2_regularization 0.0001 \
  > train_tgn_l2_0.0001.log 2>&1 &