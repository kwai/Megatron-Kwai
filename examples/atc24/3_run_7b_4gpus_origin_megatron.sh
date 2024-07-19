#!/bin/bash

# Copyright (c) 2024, Kuaishou Technology. All rights reserved.

set -euo pipefail

source ./llama-7b

export SEQ_LENGTH=12288
export GLOBAL_BATCH_SIZE=16

export HOSTFILE=
export MASTER_ADDR=127.0.0.1
export NUM_GPUS=4

export TP=1
export CP=1
export PP=4
export PP_l=1
export CKPT=full

./pretrain_llama_using_origin_megatron.sh
