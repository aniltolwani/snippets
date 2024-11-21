# Neural Snippet Recommender 🔍

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A neural search system for finding relevant code snippets and documentation. Built using a fine-tuned distilbert and trained on a dataset of all nearly all arxiv papers.

## Background 🎯

The goal is to obtain embeddings representing paragraphs, where proximity indicates likelihood of sequential paragraphs. This mechanism is used for RAG and neural search.

## Model Architecture 🏗️

The system uses a bi-encoder transformer architecture:

- Base Model: DistilBERT 
- Embedding Dimension: 768
- Optimizations:
  - Gradient checkpointing
  - Layer normalization
  - Dropout regularization
  - L2 normalized embeddings

Training includes:
- Contrastive learning with InfoNCE loss
- Mixed precision training
- Gradient accumulation
- Learning rate scheduling
- Modular config system using Hydra
- Tensorboard integration for monitoring

## Implementation Details ⚙️

- **Dataset:** Uses well-parsed articles from arxiv

- **Training Approach:**
  - Positive pairs from adjacent paragraphs (i.e. if a is before b, then (a, b) is a positive pair)
  - "Hard" negative pairs from same/other articles
  - Batched tokenization for memory efficiency
  - In-batch negatives for efficient training
  - Loss function based on cross-entropy in both row and column directions

- **Dataset Processing:**
  - Processes lists of paragraphs and article lengths
  - Stores coordinates of sampled query paragraphs
  - Handles tokenization in batches
