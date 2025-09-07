#!/bin/bash
export PYTHONPATH=$PYTHONPATH:~/workplace/python/DCRec/
nohup python main.py \
    --model tgsrec \
    --data ml-100k \
    --prefix train-tgsrec \
    --n_neg 1 \
    --n_layers 2 \
    --n_test_neg 100 \
    > train_tgsrec.log 2>&1 &