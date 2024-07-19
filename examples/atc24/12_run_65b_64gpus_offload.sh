#!/bin/bash

# Copyright (c) 2024, Kuaishou Technology. All rights reserved.

set -euo pipefail

source ./llama-65b

export SEQ_LENGTH=32768
export GLOBAL_BATCH_SIZE=16

export HOSTFILE=/etc/mpi/hostfile
export MASTER_ADDR=$MY_NODE_IP
export NUM_GPUS=64

export TP=4
export CP=2
export PP=4
export PP_l=1
export CKPT=no
export OFFLOAD_ALPHA=0.75

./pretrain_llama_using_origin_megatron.sh
