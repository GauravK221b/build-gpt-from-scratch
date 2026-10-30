# Build a GPT from Scratch with PyTorch

A learning project implementing character-level language models in PyTorch

## Overview

This repository contains two implementations:

- **`bigram.py`** — a simple bigram language model. It learns to predict the next character using a learned token-to-logit lookup table.
- **`gpt.py`** — a small character-level GPT-style Transformer with token and positional embeddings, masked self-attention, multi-head attention, feed-forward layers, residual connections, layer normalization, and dropout.

Both scripts train on `input.txt`, use a character-level vocabulary, print training/validation loss periodically, and generate text after training.

## Repository structure

```text
build-gpt-from-scratch/
├── bigram.py
├── gpt.py
├── input.txt
├── sample_output.txt
├── requirements.txt
├── .gitignore
└── README.md
```

- `input.txt`: text corpus used for training.
- `sample_output.txt`: sample generated text saved from an experiment; generated text may be incoherent, especially with a small dataset or limited training.

## Requirements

- Python 3.10+ recommended
- PyTorch
- Optional NVIDIA GPU with a compatible CUDA-enabled PyTorch installation

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/build-gpt-from-scratch.git
cd build-gpt-from-scratch
```

Replace `YOUR-USERNAME` with your GitHub username.

### 2. Create and activate a virtual environment

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you want GPU acceleration, install the PyTorch build appropriate for your CUDA setup using the official PyTorch installation selector: https://pytorch.org/get-started/locally/

## Run the models

Run commands from the repository root so the scripts can find `input.txt`.

### Train the bigram model

```bash
python bigram.py
```

### Train the GPT-style Transformer

```bash
python gpt.py
```

The GPT model is significantly larger and may train slowly on a CPU. If needed, reduce `batch_size`, `block_size`, `max_iters`, `n_embd`, `n_head`, and `n_layer` near the top of `gpt.py`. Keep `n_embd` divisible by `n_head`.

## Concepts explored

- Character-level tokenization and vocabulary creation
- Encoding characters as integer IDs and decoding IDs back to text
- Training/validation split
- Mini-batch sampling for next-character prediction
- Cross-entropy loss and the AdamW optimizer
- Autoregressive text generation
- Causal (masked) self-attention
- Multi-head attention
- Positional embeddings
- Transformer blocks, residual connections, and layer normalization

## Learning notes

The bigram model is a useful baseline: it predicts the next character from the current token representation and does not use the full context in the way a Transformer does. The GPT-style model adds attention and contextual representations so predictions can depend on preceding characters within the context window.

This is an educational, character-level model—not a production-ready chatbot or a reproduction of the scale of commercial LLMs. Results depend on the dataset, hyperparameters, hardware, and training duration.

## Credits and attribution

This project was created while following Andrej Karpathy's educational video:

- Video: https://www.youtube.com/watch?v=kCc8FmEb1nY
- The code structure and implementation are based on concepts demonstrated in that tutorial. This repository is intended for learning and experimentation.

If you extend this project, consider documenting your own changes, experiments, training time, hardware, and results here.
