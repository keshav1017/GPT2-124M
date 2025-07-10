#  GPT2 124M

## Overview

This repository contains a **PyTorch-based implementation** of GPT-2 124M model, including full training and evaluation loops, distributed training support, and integration with the HellaSwag benchmark. The code is designed for both research and practical usage, supporting single- and multi-GPU setups, efficient data loading, and logging.

## Table of Contents

- [GPT2 124M](#gpt2-124m)
  - [Overview](#overview)
  - [Table of Contents](#table-of-contents)
  - [Key Features](#key-features)
  - [File Structure](#file-structure)
  - [Model Architecture](#model-architecture)
  - [Training Pipeline](#training-pipeline)
  - [Distributed Training](#distributed-training)
  - [Data Loading](#data-loading)
  - [Evaluation](#evaluation)
  - [Logging \& Checkpointing](#logging--checkpointing)
  - [Usage](#usage)
    - [Simple Launch (Single GPU/CPU)](#simple-launch-single-gpucpu)
    - [Distributed Launch (Multi-GPU)](#distributed-launch-multi-gpu)
  - [Customization](#customization)
  - [Dependencies](#dependencies)
  - [Notes](#notes)

## Key Features

- **Fully functional GPT architecture** (embedding, multi-head self-attention, MLP, layer normalization, weight sharing)
- **Distributed Data Parallel (DDP) support** for multi-GPU training
- **Gradient accumulation** for large effective batch sizes
- **Cosine learning rate scheduler** with warmup
- **Efficient data loading** from sharded datasets
- **Validation and HellaSwag evaluation**
- **Text generation** with top-k sampling
- **Logging and checkpointing** for training progress and model recovery

## File Structure

- **Model Definition**: `GPTConfig`, `CasualSelfAttention`, `MLP`, `Block`, `GPT`
- **Data Loading**: `DataLoaderLite`, `load_tokens`
- **Training Loop**: Main script section, including optimizer setup and scheduler
- **Evaluation**: Validation loss, HellaSwag accuracy, and text generation
- **Utilities**: Helper functions for distributed setup, logging, and learning rate scheduling

## Model Architecture

- **GPTConfig**: Holds model hyperparameters (block size, vocab size, layers, heads, embedding size).
- **CasualSelfAttention**: Implements multi-head self-attention with causal masking and flash attention for efficiency.
- **MLP**: Two-layer feedforward network with GELU activation.
- **Block**: Transformer block combining attention, MLP, and layer normalization with residual connections.
- **GPT**: Top-level model, stacking multiple blocks, applying embeddings, and sharing weights between input and output layers. Includes methods for initialization, forward pass, loading pretrained weights, and optimizer configuration.

## Training Pipeline

1. **Distributed Setup**: Initializes DDP if launched with `torchrun`, otherwise defaults to single-process mode.
2. **Random Seeds**: Sets deterministic behavior for reproducibility.
3. **Data Loaders**: Instantiates `DataLoaderLite` for training and validation splits, supporting sharded datasets and distributed sampling.
4. **Model Creation**: Builds the GPT model and moves it to the appropriate device.
5. **Optimizer**: Uses AdamW with optional fused kernel for CUDA, separates decayed and non-decayed parameter groups.
6. **Learning Rate Scheduler**: Linear warmup followed by cosine decay.
7. **Training Loop**:
   - Loads batches, accumulates gradients, and performs optimizer steps.
   - Periodically evaluates validation loss and HellaSwag accuracy.
   - Generates sample text sequences at intervals.
   - Logs metrics and saves checkpoints at defined steps.

## Distributed Training

- **Initialization**: Uses environment variables (`RANK`, `LOCAL_RANK`, `WORLD_SIZE`) to set up DDP with NCCL backend.
- **Device Assignment**: Each process is assigned to a specific GPU.
- **Synchronization**: Reduces validation loss and evaluation metrics across all processes.
- **Checkpointing**: Only the master process handles logging and saving checkpoints.

## Data Loading

- **DataLoaderLite**:
  - Loads sharded `.npy` token files from a specified directory.
  - Supports both training and validation splits.
  - Handles data partitioning for distributed processes.
  - Provides batched input/target pairs for language modeling.

## Evaluation

- **Validation Loss**: Computed every 250 steps (or at last step), averaged over several batches.
- **HellaSwag Benchmark**: Evaluates multiple-choice completion accuracy using a masked loss.
- **Text Generation**: Samples sequences from the model using top-k sampling and prints outputs for inspection.

## Logging & Checkpointing

- **Logging**: Writes step-wise metrics (loss, accuracy) to a log file.
- **Checkpoints**: Periodically saves model state, configuration, training step, and validation loss for recovery or later analysis.

## Usage

### Simple Launch (Single GPU/CPU)

```bash
python train.py
```

### Distributed Launch (Multi-GPU)

```bash
torchrun --standalone -nproc_per_node=8 train.py
```

- Adjust `nproc_per_node` to match the number of GPUs.

## Customization

- **Model Size**: Change `GPTConfig` parameters (layers, heads, embedding size) for different model scales.
- **Batch Size & Sequence Length**: Modify `total_batch_size`, `B`, and `T` as needed.
- **Learning Rate Schedule**: Tune `max_lr`, `min_lr`, `warmup_steps`, and `max_steps`.
- **Data Directory**: Set `data_root` in `DataLoaderLite` to point to your dataset location.

## Dependencies

- Python 3.8+
- PyTorch (with CUDA for GPU support)
- NumPy
- tiktoken
- transformers (for loading HuggingFace GPT-2 weights)
- HellaSwag dataset utilities

Install dependencies via pip:

```bash
pip install torch numpy tiktoken transformers
```

## Notes

- **DDP requires CUDA**: Distributed training only works with GPUs.
- **Data Format**: Expects tokenized `.npy` files in the `edu_fineweb10B` directory, split into `train` and `val` shards.
- **Precision**: Uses `bfloat16` autocasting for efficiency on supported hardware.
- **Checkpoints**: For full reproducibility, consider saving optimizer state and RNG seeds.
- **Text Generation**: The script prints generated samples every 250 steps for qualitative monitoring.

For further customization or troubleshooting, refer to the inline comments throughout the code. This script is intended as a research and experimentation framework for GPT-like models, and can be extended for more advanced use cases.