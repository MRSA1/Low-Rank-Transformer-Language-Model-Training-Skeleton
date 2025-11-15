#!/bin/bash
# Pretraining script for Low-Rank Transformer

python train.py train \
    --preset base17 \
    --source "HuggingFaceFW/fineweb-edu,togethercomputer/RedPajama-Data-1T,oscar-corpus/OSCAR-2201:en" \
    --block 576 \
    --steps 50000 \
    --amp \
    --auto_grow \
    --grow_plan "576,768,1024,1536" \
    --grow_every_steps 10000 \
    --save_dir ./models/pretrain-base17 \
    --save_every_sec 7200 \
    --lr_core 5e-5 \
    --lr_head 2e-4
