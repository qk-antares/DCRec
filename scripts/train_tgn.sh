#!/bin/bash
export PYTHONPATH=$PYTHONPATH:~/workplace/python/DCRec/
nohup python main.py \
  --use_memory \
  --prefix tgn \
  --n_neg 10 \
  > train_tgn.log 2>&1 &