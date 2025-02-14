"""
Testing script for fine-tuned DeepSeek-R1-Distill-Llama-8B model.
This script loads and tests a saved model on medical reasoning tasks.
"""

import os
import torch
import platform
import warnings
from transformers import logging        # type: ignore

from utils import (                     # type: ignore
    get_model, get_tokenizer,
    get_pipeline,
)
from config import config

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
logging.set_verbosity_error()

def test_saved_model(model_path="./fine_tuned_model"):
    """Test the saved fine-tuned model with memory optimizations for Apple Silicon."""
    print("\nLoading model from:", model_path)

    # Create offload directory
    os.makedirs("offload", exist_ok=True)

    try:
        # Load model with proper memory management for Apple Silicon
        kwargs = {
            "offload_folder": "offload",
            "offload_state_dict": True
        }
        model = get_model(model_path, offload_folder="offload", offload_state_dict=True)

        tokenizer = get_tokenizer(model_path)

        # Initialize pipeline with optimized settings
        pipe = get_pipeline(model, tokenizer)

        # Test cases
        test_cases = [
            """Please reason step by step:

A 45-year-old patient presents with sudden onset chest pain, shortness of breath, and anxiety. The pain is described as sharp and worsens with deep breathing. What is the most likely diagnosis and what immediate tests should be ordered?""",

            """Please reason step by step:

A 67-year-old diabetic patient presents with sudden onset of severe headache, confusion, and right-sided weakness. What is the most likely diagnosis and what immediate imaging should be performed?"""
        ]

        print("\nRunning test cases...")
        for i, test_case in enumerate(test_cases, 1):
            print(f"\nTest Case {i}:")
            print("Input:", test_case)
            print("\nGenerating response...")

            result = pipe(
                test_case,
                max_new_tokens=512,
                temperature=0.6,
                top_p=0.95,
                repetition_penalty=1.15
            )

            print("\nModel Response:", result[0]["generated_text"])
            print("\n" + "="*80)

    except Exception as e:
        print(f"\nError during testing: {str(e)}")
    finally:
        # Cleanup
        if os.path.exists("offload"):
            import shutil
            shutil.rmtree("offload")

if __name__ == "__main__":
    if not os.path.exists("./fine_tuned_model"):
        print("Error: Could not find fine-tuned model at './fine_tuned_model'")
        print("Please ensure the model was saved correctly during training.")
    else:
        test_saved_model()
