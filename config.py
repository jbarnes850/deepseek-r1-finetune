"""Global configuration settings for the DeepSeek fine-tuning project."""

import os
from dataclasses import dataclass

@dataclass
class Config:
    # Model settings
    MODEL_NAME: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    LEARNING_RATE: float = 1e-5
    BATCH_SIZE: int = 2
    NUM_EPOCHS: int = 1

    # Dataset settings
    DEFAULT_DATASET: str = "medical-o1-reasoning-SFT"
    DATASET_SPLIT_RATIO: float = 0.05  # Use 5% of data for faster tutorial run

    # LoRA settings
    LORA_RANK: int = 8
    LORA_ALPHA: int = 16
    LORA_DROPOUT: float = 0.1

    # Training settings
    OUTPUT_DIR: str = "deepseek-r1-medical-finetuning"
    WANDB_PROJECT: str = "deepseek-r1-medical-finetuning"

    # System settings
    CACHE_DIR: str = os.path.join(os.path.expanduser("~"), ".cache/huggingface")

# Create a global instance
config = Config()
