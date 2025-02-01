# DeepSeek R1 Fine-tuning for Apple Silicon

This repository contains code for fine-tuning the DeepSeek R1 Distill Llama 8B model on Apple Silicon (M-series) machines. The implementation is optimized for M1/M2/M3 with at least 16GB RAM with Apple Silicon-specific optimizations.

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

## Troubleshooting

### Common Warnings and Solutions

1. **Tokenizer Deprecation Warning**
   - This warning has been resolved by using the proper data collator and processing configuration
   - The script now uses the recommended approach for SFTTrainer

2. **bitsandbytes Warning**
   - This warning is expected on Apple Silicon as bitsandbytes GPU support is not needed
   - The script is optimized to use native MPS acceleration instead

3. **Memory-Related Warnings**
   - The script uses optimized memory settings for Apple Silicon
   - Gradient checkpointing and proper batch sizes are configured
   - KV-cache is disabled to prevent memory issues

### Error Prevention

The script includes several optimizations to prevent common errors:

- Proper warning suppression for cleaner output
- Optimized tokenizer and data collator configuration
- Memory-efficient training settings
- Apple Silicon-specific optimizations

If you encounter any issues:

1. Ensure you're using Python 3.11+ on Apple Silicon
2. Verify all dependencies are correctly installed
3. Check available system memory (32GB+ recommended)
4. Monitor training through W&B dashboard for detailed metrics
