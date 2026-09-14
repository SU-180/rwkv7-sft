#!/bin/bash

export PYTHONNOUSERSITE=1

# 禁止 transformers 引入 tensorflow
export USE_TF=0
export USE_FLAX=0

export CUDA_VISIBLE_DEVICES=0,1

accelerate launch \
--num_processes=2 \
scripts/train_rwkv7_qlora.py
