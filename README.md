# DeepSeek R1 Fine-tuning for Apple Silicon

This repository contains code for fine-tuning the DeepSeek R1 Distill Llama 8B model on Apple Silicon (M-series) machines. The implementation is optimized for the M3 Max with 36GB RAM and follows the [DataCamp tutorial](https://www.datacamp.com/tutorial/fine-tuning-deepseek-r1-reasoning-model) with Apple Silicon-specific optimizations.

## Features

- Optimized for Apple Silicon using MPS (Metal Performance Shaders)
- Memory-efficient training with gradient checkpointing and mixed precision
- Weights & Biases integration for experiment tracking
- GSM8K dataset for mathematical reasoning
- LoRA fine-tuning for efficient model adaptation
- Base configuration optimized for learning and demonstration purposes (2-3 hour training time)

## Project Structure

- `deepseek_finetune.py`: Main Python script for fine-tuning
- `requirements.txt`: List of Python dependencies
- `README.md`: Project documentation

## Setup Instructions

1. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Log in to Weights & Biases:

```bash
wandb login
```

4. Run the fine-tuning script:

```bash
python deepseek_finetune.py
```

## Prerequisites

- Python 3.11+
- Apple Silicon Mac (M1/M2/M3)
- At least 16GB RAM (32GB+ recommended)
- Weights & Biases account for experiment tracking

## Model Configuration

The script includes several optimizations for Apple Silicon:

- MPS device mapping for hardware acceleration
- Mixed precision training (FP16)
- Gradient checkpointing for memory efficiency
- Small batch size with increased gradient accumulation
- LoRA fine-tuning for parameter efficiency

## Training Parameters

- Model: DeepSeek R1 Distill Llama 8B
- Batch size: 1 (with gradient accumulation steps of 16)
- Learning rate: 2e-4
- Training epochs: 1
- LoRA rank: 8
- LoRA alpha: 16

## Monitoring Training

Training progress can be monitored through:

1. Terminal output showing loss and training steps
2. Weights & Biases dashboard for detailed metrics
3. Saved model checkpoints in `./results` directory

## Output

The fine-tuned model will be saved in `./fine_tuned_model` directory and can be loaded using the Hugging Face Transformers library.

## Acknowledgments

- Based on the [DataCamp DeepSeek R1 fine-tuning tutorial](https://www.datacamp.com/tutorial/fine-tuning-deepseek-r1-reasoning-model)
- Uses the [DeepSeek R1 Distill Llama 8B model](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B)
