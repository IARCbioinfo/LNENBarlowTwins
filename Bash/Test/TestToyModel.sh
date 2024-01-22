#!/bin/bash
python ../../main.py \
    --evaluate \
    --list-dir ../../Datasets/ToyTestSetKi67Tumor.txt \
    --projector 1024-512-256-128 \
    --checkpoint_evaluation ../../Checkpoint_toy_LNEN/checkpoint_43.pth \
    --checkpoint-dir ../../Checkpoint_toy_LNEN \
    --projector-dir ../../projector_toy \
    --batch-size 12