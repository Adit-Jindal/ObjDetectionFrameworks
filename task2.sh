#!/bin/bash

EXP=$1   # ce or fl
MODE=$2  # train
DATA=$3

if [ "$MODE" = "train" ]; then
    python task2/train.py $DATA $EXP
fi