#!/bin/bash
export PYTHONPATH=$PYTHONPATH:~/workplace/python/DCRec/
nohup python main.py \
    --train_ratio 0.05 \
    --valid_ratio 0.01 \
    --prefix tgsrec-test \
    --n_neg 1 \
    --n_layers 2 \
    --n_epoch 3 \
    --n_skip_val 0 \
    --n_test_neg 100 \
    > test_tgsrec.log 2>&1 &