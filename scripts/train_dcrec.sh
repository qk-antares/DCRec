#!/bin/bash
export PYTHONPATH=$PYTHONPATH:~/workplace/python/DCRec/
cd ~/workplace/python/DCRec/
nohup python main.py \
  --use_memory \
  --prefix dcrec \
  --n_neg 10 \
  > train_dcrec.log 2>&1 &