# Artifact Evaluation for USENIX ATC '24

This directory contains scripts used to reproduce the results in "*Accelerating the Training of Large Language Models using Efficient Activation Rematerialization and Optimal Hybrid Parallelism*" that is to appear at USENIX ATC '24. These scripts use OpenMPI, but can be modified for other schedulers as well.

## Getting Started Instructions

### Get Commit

- **Code location:** https://github.com/kwai/Megatron-Kwai/tree/atc24ae/examples/atc24
- **Submitted tag:** atc24ae-1.0.0

```bash
git clone https://github.com/kwai/Megatron-Kwai.git --branch atc24ae ~/Megatron-Kwai
```

The referred "Latest Megatron-LM" is the snapshot of NVIDIA/Megatron-LM at Jan 1, 2024, when the commit id was [2bc6cd3](https://github.com/NVIDIA/Megatron-LM/commit/2bc6cd307a11423928c675f741e79e03df23e721). We make minor modifications (&#126;10 lines) on it to ensure compatibility with OpenMPI and for our dataset.

```bash
git clone https://github.com/kwai/Megatron-Kwai.git --branch jan-1-2024-main ~/Megatron-LM-Jan-1-2024
```

### Hardware Requirements

Minimum requirements

- Purpose: To reproduce the accelerating method on a minimum demo.
- Hardware: One node (i.e., a server) equipped with four NVIDIA A100/A800/H100/H800 80GB GPU cards.

### Software Requirements

Software requirements are consistent with NVIDIA's official Megatron-LM. The only additional requirement is TransformerEngine [v1.1.0](https://github.com/NVIDIA/TransformerEngine/tree/v1.1) built with `NVTE_WITH_USERBUFFERS=1`.

To reproduce the performance, it is suggested to use the same software version as specified in the paper Section 6.1 "Experimental Settings". We also provide a Docker image that complies with all the suggested software.

```bash
docker pull yuantailing/megatron-kwai:atc24ae-1.0.0
```

### Dataset

A subset (&#126;100M tokens) of the [enwiki dataset](https://dumps.wikimedia.org/enwiki/) is located at `/root/dataset` within the Docker image. The scripts can be modified for other dataset as well.

### Scripts (Minimum Demo)

Train Llama-7B with a context window size of 12,288 using 4 GPUs.

```bash
cd ~/Megatron-Kwai/examples/atc24
./1_run_7b_4gpus_baseline.sh
./2_run_7b_4gpus_ours.sh
./3_run_7b_4gpus_origin_megatron.sh
./4_run_7b_4gpus_no_ckpt_oom.sh
```

Time estimation: Initialization for each script requires less than 1 minute. Each iteration takes approximately 5-10 seconds on H800, or 10-20 seconds on A800. For the first run, an additional 1-3 minutes may be needed to create the dataset index and compile the CUDA extensions.

The expected performance is listed as follows.

| File | Method | A800 throughput / MFU | H800 throughput / MFU |
|--|--|--|--|
| 1_run_7b_4gpus_baseline.sh | baseline | 3016 / 47.7% | 6639 / 33.1% |
| 2_run_7b_4gpus_ours.sh | ours | 3803 / 60.1% | 8273 / 41.2% |
| 3_run_7b_4gpus_origin_megatron.sh | origin Megatron-LM | 2847 / 45.0% | 6138 / 30.6% |
| 4_run_7b_4gpus_no_ckpt_oom.sh | baseline w/o full ckpt | OOM | OOM |

The throughput metric is "Tokens per Second per GPU" printed to the screen.

Comment: The purpose of the minimum demo is to check that our activation rematerialization mechanism works, but pipeline parallelism may not be the best configuration for training Llama-7B s=12k.

### Artifact Claims

1. Comparing script 1 and script 2: Our method achieves a significant performance improvement over the baseline by utilizing offloading and balanced checkpointing.
2. Comparing script 2 and script 4: Our activation rematerialization mechanism reduces GPU memory consumption.
3. Comparing script 1 and script 3: The baseline is stronger than the origin Megatron-LM.
4. Comparing script 1 and script 4: The baseline method raises out-of-memory error (OOM) if full checkpointing is not used, thus `CKPT=full` is essential when our method is not in use.
5. Comparing script 2 and script 3: The loss curves are similar, suggesting the correctness of our techniques. Comment: The loss curves cannot be identical due to different initialization of models parameters.

## Detailed Instructions

### Hardware Requirements

Recommended requirements

- Purpose: To reproduce the exact performance reported in the paper.
- Hardware: A cluster consists of 32 nodes. Each node is equipped with eight NVIDIA H800 80GB GPUs interconnected via NVLink. For inter-node communication, each node is outfitted with eight 100 Gbps NICs. Each node is configured with two CPUs and &ge;1TB of host memory. Each GPU is connected to a CPU via PCIe 5.0 x16.

### Dataset

The dataset should be copied to a directory that is shared by all nodes. Change `DATA_PATH` in the *pretrain_llama.sh* accordingly.

### Reproduce Table 8

Train Llama-175B with a context window size of 8,192 using 256 GPUs.

```bash
cd ~/Megatron-Kwai/examples/atc24
./5_run_175b_256gpus_baseline.sh
./6_run_175b_256gpus_ours.sh
./7_run_175b_256gpus_origin_megatron.sh
./8_run_175b_256gpus_no_ckpt_oom.sh
```

The expected performance is listed as follows.

| File | Method | H800 throughput / MFU |
|--|--|--|
| 5_run_175b_256gpus_baseline.sh | baseline | 299 / 33.4% |
| 6_run_175b_256gpus_ours.sh | ours | 387 / 43.2% |
| 7_run_175b_256gpus_origin_megatron.sh | origin Megatron-LM | 278 / 31.0% |
| 8_run_175b_256gpus_no_ckpt_oom.sh | baseline w/o full ckpt | OOM |

Note: Different clusters may require different MPI parameters. Please update the `HOSTFILE` based on the cluster's node configuration and `CLUSTER_MPI_ARGS` according to the specific MPI settings required.

Debug tips: If you encounter problems running multi-node scripts, try running the official examples from the origin Megatron-LM on multiple nodes first, ensuring that all types of parallelism -- such as Tensor Parallelism (TP), Context Parallelism (CP), Pipeline Parallelism (PP), and Data Parallelism (DP) -- are enabled. This preliminary step will help verify that the MPI arguments are correctly configured. Once that's done, the script `./7_run_175b_256gpus_origin_megatron.sh` should execute without issues, and so do other scripts. Furthermore, for the "TP overlap" feature, more precise configuration of MPI arguments is required. As a debugging step, you can temporarily remove the `TP_OVERLAP_ARGS` and see if the issue is resolved.

To reproduce the results for other rows in Table 8 (Section 6.4):
- Change `source ./llama-175b` to `source ./llama-65b` or `source ./llama2-70b` to apply the respective models.
- Change `SEQ_LENGTH=8192` variable to other sequence lengths as needed.
- Change the variables TP, CP, PP, PP_l, CKPT, and OFFLOAD_ALPHA according to the values of $t$, $c$, $p$, $l$, ckpt, and $\alpha$ as listed in the table.

### Reproduce Figure 7

To reproduce the results for Figure 7 (Section 6.2):
- Run `./9_run_65b_256gpus_verify_memory.sh`, and observe the GPU memory usage after the 2nd iteration.
- Note 1: If the GPU cards are not NVIDIA H800, the observed memory usage may vary slightly, due to the different behavior of PyTorch across GPU types.
- Note 2: `max_memory_allocated` can be exactly reproduced. Other memory usage, including `max_memory_reserved` and `memory_info.used`, may exhibit slight variations even when executing the same script in multiple times.
- Note 3: Peak memory usage should be observed after the 2nd itereation because some optimizer states are initialized at the end of 1st iteration.

The exact values of `max_memory_allocated` are listed below.

| $\alpha$ | ckpt=no `max_memory_allocated` | ckpt=ours `max_memory_allocated` |
|--|--|--|
| 0.00 | OOM | 65330982400 |
| 0.10 | OOM | 62128631296 |
| 0.20 | OOM | 58907405824 |
| 0.30 | 72762730496 | 55554059776 |
| 0.40 | 66951784960 | 52089073152 |
| 0.50 | 61136119808 | 48604098048 |
| 0.60 | 55733856256 | 45722611200 |
| 0.70 | 50331592704 | 42832735744 |
| 0.80 | 45824813056 | 38925740032 |
| 0.90 | 39527328256 | 36144916480 |
| 1.00 | 34491172352 | 32462145024 |

### Reproduce Figure 9

To reproduce the results for Figure 9 (Section 6.5):
- Change `source ./llama-175b` and modify `SEQ_LENGTH` to apply the respective models.
- The values of `NUM_GPUS`, `GLOBAL_BATCH_SIZE`, TP, CP, PP, PP_l, CKPT, and OFFLOAD_ALPHA used in the experiments are provided in the following table.

| model | `SEQ_LENGTH` | `NUM_GPUS` | $B$ | $t$ | $c$ | $p$ | $l$ | ckpt | $\alpha$ | tokens per second |
|--|--|--|--|--|--|--|--|--|--|--|
| Llama-175B | 4096 | 256 | 256 | 2 | 2 | 16 | 1 | no | 0.522 | 97577 |
| Llama-175B | 4096 | 240 | 240 | 4 | 1 | 12 | 1 | no | 0.276 | 85924 |
| Llama-175B | 4096 | 192 | 240 | 2 | 2 | 24 | 1 | no | 0.410 | 76130 |
| Llama-175B | 4096 | 160 | 240 | 4 | 1 | 8 | 2 | ours | 0.334 | 59169 |
| Llama-175B | 4096 | 144 | 252 | 4 | 1 | 6 | 2 | ours | 0.746 | 56637 |
| Llama-175B | 4096 | 128 | 256 | 2 | 1 | 16 | 1 | ours | 0.751 | 54968 |
| Llama-175B | 4096 | 120 | 270 | 4 | 1 | 6 | 2 | ours | 0.851 | 45761 |
| Llama-175B | 4096 | 96 | 264 | 4 | 1 | 12 | 2 | ours | 0.324 | 41048 |
| Llama-175B | 4096 | 64 | 272 | 4 | 1 | 8 | 1 | ours | 0.979 | 27708 |
| Llama-175B | 4096 | 48 | 264 | 4 | 1 | 12 | 1 | ours | 0.999 | 20685 |
| Llama-65B | 4096 | 256 | 256 | 2 | 1 | 8 | 2 | no | 0.355 | 233981 |
| Llama-65B | 4096 | 240 | 240 | 2 | 1 | 10 | 2 | no | 0.302 | 219415 |
| Llama-65B | 4096 | 200 | 250 | 4 | 1 | 5 | 4 | no | 0.000 | 179819 |
| Llama-65B | 4096 | 192 | 240 | 4 | 1 | 8 | 2 | no | 0.000 | 180303 |
| Llama-65B | 4096 | 160 | 240 | 2 | 1 | 10 | 2 | no | 0.331 | 158984 |
| Llama-65B | 4096 | 128 | 256 | 2 | 1 | 8 | 2 | ours | 0.000 | 132357 |
| Llama-65B | 4096 | 120 | 240 | 2 | 1 | 5 | 2 | ours | 0.383 | 121488 |
| Llama-65B | 4096 | 112 | 252 | 4 | 1 | 4 | 4 | no | 0.000 | 99046 |
| Llama-65B | 4096 | 96 | 240 | 2 | 1 | 8 | 2 | ours | 0.035 | 96758 |
| Llama-65B | 4096 | 80 | 240 | 2 | 1 | 10 | 2 | ours | 0.000 | 87396 |
| Llama-65B | 4096 | 64 | 256 | 2 | 1 | 8 | 2 | ours | 0.162 | 71884 |
| Llama-65B | 4096 | 56 | 252 | 2 | 1 | 4 | 2 | ours | 0.945 | 52219 |
| Llama-65B | 4096 | 48 | 264 | 2 | 1 | 8 | 2 | ours | 0.290 | 51124 |
| Llama-65B | 4096 | 40 | 260 | 2 | 1 | 5 | 2 | ours | 0.815 | 45716 |
| Llama-65B | 4096 | 32 | 272 | 2 | 1 | 8 | 2 | ours | 0.544 | 36740 |
| Llama2-70B | 8192 | 256 | 256 | 2 | 4 | 8 | 2 | no | 0.000 | 229338 |
| Llama2-70B | 8192 | 240 | 270 | 2 | 4 | 10 | 2 | no | 0.000 | 217316 |
| Llama2-70B | 8192 | 224 | 252 | 2 | 4 | 4 | 4 | ours | 0.352 | 192634 |
| Llama2-70B | 8192 | 200 | 250 | 2 | 2 | 5 | 2 | ours | 0.368 | 182038 |
| Llama2-70B | 8192 | 192 | 264 | 2 | 4 | 8 | 2 | no | 0.000 | 177032 |
| Llama2-70B | 8192 | 160 | 260 | 2 | 4 | 10 | 2 | no | 0.000 | 148456 |
| Llama2-70B | 8192 | 144 | 252 | 2 | 2 | 4 | 2 | ours | 0.755 | 124599 |
| Llama2-70B | 8192 | 128 | 256 | 2 | 2 | 8 | 2 | ours | 0.012 | 122265 |
| Llama2-70B | 8192 | 120 | 270 | 2 | 2 | 5 | 2 | ours | 0.460 | 114817 |
| Llama2-70B | 8192 | 112 | 252 | 2 | 2 | 4 | 2 | ours | 0.811 | 100242 |
| Llama2-70B | 8192 | 96 | 264 | 2 | 2 | 8 | 2 | ours | 0.080 | 91488 |
| Llama2-70B | 8192 | 80 | 260 | 2 | 2 | 10 | 2 | ours | 0.024 | 78698 |
| Llama2-70B | 8192 | 64 | 256 | 2 | 1 | 8 | 2 | ours | 0.654 | 65302 |
| Llama2-70B | 8192 | 48 | 264 | 2 | 1 | 8 | 2 | ours | 0.722 | 47755 |
| Llama2-70B | 8192 | 40 | 260 | 2 | 1 | 10 | 2 | ours | 0.713 | 41182 |
| Llama2-70B | 8192 | 32 | 272 | 2 | 1 | 8 | 2 | ours | 0.858 | 33262 |

### Artifact Claims

1. All the memory usage values in Figure 7 (Section 6.2) are reproduced.
2. All the "Throughput / MFU" values listed in Table 8 (Section 6.4) are reproduced.
3. All the "achieved throughput" values in Figure 9 (Section 6.5) are reproduced.
