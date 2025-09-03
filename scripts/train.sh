#!/bin/bash
export PYTHONPATH=$PYTHONPATH:~/workplace/python/DCRec/
nohup python main.py \
  --use_memory \
  --prefix tgn-attn \
  --n_neg 10 \
  > train.log 2>&1 &