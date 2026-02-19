Swin Transformer From Scratch (PyTorch)

This project is a clean, from-scratch implementation of a Swin Transformer for image classification using PyTorch.
The model is trained and evaluated on the MNIST dataset and reproduces the core architectural ideas of the original Swin Transformer paper in a simplified and educational form.

The goal of this repository is not to rely on external vision transformer libraries, but to implement the key building blocks manually and understand how hierarchical window-based attention works in practice.

Project Overview

This implementation includes:

Manual window partitioning and window reversal

Window-based Multi-Head Self-Attention

Shifted window mechanism

Patch embedding using convolution

Patch merging (hierarchical feature downsampling)

Two-stage Swin architecture

Full training and evaluation pipeline on MNIST

The entire model is implemented using native PyTorch modules.

Architecture Summary

The model follows a simplified Swin Transformer pipeline:

Patch Embedding

Uses a Conv2D layer with stride to convert the image into patch embeddings.

MNIST input (1 × 28 × 28) is converted into patch tokens.

Stage 1

Swin Blocks with window-based attention.

Includes shifted and non-shifted attention.

Patch Merging

Reduces spatial resolution.

Doubles the feature dimension.

Stage 2

Another sequence of Swin Blocks operating on merged patches.

Classification Head

Global pooling.

Fully connected layer for 10-class classification.

This mirrors the hierarchical design principle of Swin Transformers.

Core Components
1. Window Partitioning

Splits feature maps into non-overlapping windows for local attention computation.

2. Window Reverse

Reconstructs the original spatial layout after attention.

3. WindowAttention

Implements multi-head self-attention inside each window.

4. SwinBlock

Contains:

LayerNorm

Window-based attention

Optional shifted windows

MLP

Residual connections

5. PatchMerging

Combines neighboring patches and increases embedding dimension.

6. TwoStageSwinMNIST

Full model composed of:

Patch embedding

Two Swin stages

Classification head

Dataset

MNIST

Automatically downloaded via torchvision

Preprocessing:

ToTensor

Normalization with MNIST mean and std

Training Setup

Optimizer: Adam

Loss: CrossEntropyLoss

Epochs: 5 (configurable)

Device: CUDA if available, otherwise CPU

The model achieves strong classification performance on MNIST within a few epochs.

Installation

Clone the repository:

git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name


Install dependencies:

pip install torch torchvision tqdm

How to Run

If using the notebook:

jupyter notebook Swin_transformers_from_scratch.ipynb


Run all cells sequentially.

If converting to a Python script:

python train.py

Why This Project Matters

This repository demonstrates:

Understanding of Vision Transformer internals

Implementation of hierarchical attention mechanisms

Knowledge of shifted window strategy

Ability to build transformer-based vision models without external transformer frameworks

This is not a wrapper around an existing model. The attention mechanism, window logic, and architecture are implemented manually.

Limitations

Designed for MNIST only (28×28 grayscale images)

Simplified compared to the original Swin Transformer

No relative positional bias

No large-scale training experiments

This is an educational and foundational implementation, not a production-ready ImageNet-scale model.

Possible Improvements

Add relative position bias

Extend to CIFAR-10 or CIFAR-100

Add configurable depth and number of heads

Implement full multi-stage Swin architecture

Add training logs and visualization

Convert notebook into modular production-ready code

Repository Structure
Swin_transformers_from_scratch.ipynb
README.md

Author

Hamid Gholami
Computer Vision and LLM Engineer
Focus: Vision Transformers, LLMs, RAG Systems, Generative AI
