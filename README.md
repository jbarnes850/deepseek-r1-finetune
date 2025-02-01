# DeepSeek R1 Fine-tuning for Apple Silicon

This repository demonstrates how to fine-tune the DeepSeek-R1-Distill-Llama-8B model for medical reasoning tasks on Apple Silicon (M1/M2/M3) Macs. The implementation is optimized for machines with 16GB+ RAM and includes both training and testing workflows.

## Overview

- **Model**: [DeepSeek-R1-Distill-Llama-8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B)
- **Dataset**: [Medical Reasoning Dataset](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT)
- **Training Time**: ~2-3 hours (tutorial configuration)
- **Hardware**: Apple Silicon M1/M2/M3 with 16GB+ RAM

## Project Structure

```bash
.
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── deepseek_finetune.py        # Training script
├── test_model.py               # Testing script
└── .gitignore                  # Git ignore file
```

## Features

- Optimized for Apple Silicon using MPS (Metal Performance Shaders)
- Memory-efficient training with gradient checkpointing
- LoRA fine-tuning for efficient model adaptation
- Weights & Biases integration for experiment tracking
- Separate testing script for model evaluation
- Tutorial-optimized configuration (5% dataset, 2-3 hour training)

## Setup Instructions

1. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Log in to Weights & Biases:

```bash
wandb login
```

## Training Process

1. Run the fine-tuning script:

```bash
python deepseek_finetune.py
```

The training script includes:

- Dataset preparation with DeepSeek's recommended prompt format
- Memory-efficient model configuration
- LoRA training setup
- Progress tracking via W&B
- Model saving

### Training Parameters

```python
# Dataset
dataset_size = 0.05  # 5% of dataset (~1,268 examples)
max_length = 1024    # Sequence length

# Training
batch_size = 2
gradient_accumulation_steps = 4
learning_rate = 1e-4
num_epochs = 1

# LoRA Parameters
lora_r = 4          # LoRA rank
lora_alpha = 16
lora_dropout = 0.1
```

### Expected Training Metrics

1. **Initial Loss**: ~1.8-2.0
2. **Training Progression**:
   - First 25% of steps: Rapid decrease to ~1.6
   - Middle 50% of steps: Gradual decline to ~1.4
   - Final 25%: Stabilization around 1.3-1.4
3. **Target Loss**: 1.2-1.4 (indicates successful adaptation)

## Testing Process

After training completes, test your model:

```bash
python test_model.py
```

The testing script includes:

- Memory-efficient model loading
- Multiple medical reasoning test cases
- Proper prompt formatting
- Detailed response generation

### Testing Parameters

```python
# Generation Settings
max_new_tokens = 512
temperature = 0.6      # DeepSeek recommended
top_p = 0.95
repetition_penalty = 1.15
```

## Memory Management

Both scripts include optimizations for Apple Silicon:

- Gradient checkpointing during training
- Model offloading during testing
- Efficient tokenization
- Proper memory cleanup

## Monitoring

1. **During Training**:
   - Terminal output shows loss every 10 steps
   - W&B dashboard tracks metrics
   - Model checkpoints saved after each epoch

2. **During Testing**:
   - Terminal output shows generated responses
   - Memory usage monitoring
   - Proper error handling

## Common Issues and Solutions

1. **Out of Memory Errors**:
   - Reduce batch size
   - Decrease sequence length
   - Enable gradient checkpointing
   - Reduce dataset size

2. **Slow Training**:
   - Increase learning rate
   - Reduce dataset size
   - Decrease sequence length
   - Adjust gradient accumulation

3. **Poor Loss Convergence**:
   - Increase training epochs
   - Adjust learning rate
   - Increase LoRA rank
   - Use larger dataset portion

## Advanced Configuration

For production use, consider:

```python
# Dataset
dataset_size = 0.2    # at least 20% of dataset
max_length = 2048     # Full sequence length

# Training
batch_size = 1
gradient_accumulation_steps = 8
learning_rate = 5e-5
num_epochs = 2

# LoRA Parameters
lora_r = 8           # Higher rank
```

## References

- [DeepSeek Model Card](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B)
- [Medical Dataset Documentation](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
