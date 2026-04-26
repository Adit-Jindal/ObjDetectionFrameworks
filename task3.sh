#!/bin/bash

EXP=$1   # pre / full / decoder / encoder
MODE=$2
DATA=$3

if [ "$MODE" = "evaluate" ]; then
    python task3/eval.py $DATA
else
    python task3/train.py $DATA $EXP
fi