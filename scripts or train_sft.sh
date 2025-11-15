#!/bin/bash
# Instruction tuning script

python train.py train \
    --preset base17 \
    --warmstart_from ./models/pretrain-base17/final.pt \
    --after_sft_steps 20000 \
    --after_sft_source "mlabonne/opc-sft-stage2-chat,HuggingFaceH4/ultrachat_200k" \
    --after_sft_chat \
    --after_sft_block 1120 \
    --after_sft_lr_core 1e-5 \
    --after_sft_lr_head 5e-5 \
    --save_dir ./models/sft-base17
