"""
Fine-tuning script for DeepSeek-R1-Distill-Llama-8B model on Apple Silicon.
This tutorial demonstrates how to fine-tune the model for medical reasoning tasks.
Optimized for M2/M3 Macs with ~36GB RAM.

References:
- Model: https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B
- Dataset: https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT
"""

import token
from regex import template
import torch
import platform
import warnings
import wandb
import os
from transformers import (          # type: ignore
    TrainingArguments,
    logging,
    DataCollatorForLanguageModeling,
    AutoTokenizer,
    AutoModelForCausalLM,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
from datasets import load_dataset# type: ignore

from utils import get_model, get_pipeline, get_tokenizer
from test_model import test_saved_model
from config import config
import preprocess

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.set_verbosity_error()


def do_wandb_init():
    # Initialize wandb for experiment tracking
    wandb.init(
        project="deepseek-r1-medical-finetuning",
        config={
            "model_name": config.MODEL_NAME,
            "learning_rate": config.LEARNING_RATE,
            "batch_size": config.BATCH_SIZE,
            "num_epochs": config.NUM_EPOCHS,
            "hardware": "Apple Silicon",
            "dataset": config.DEFAULT_DATASET,
            "lora_rank": config.LORA_RANK  ,
            "lora_alpha": config.LORA_ALPHA
        }
    )

# Initialize device for Apple Silicon
if platform.processor() == 'arm' and torch.backends.mps.is_available():
    print("Using Apple Silicon MPS (Metal Performance Shaders)")
    device = torch.device("mps")
elif torch.cuda.is_available():
    print("Using CUDA GPU")
    device = torch.device("cuda")
else:
    print("Using CPU")
    device = torch.device("cpu")

def setup_model():
    """Setup the DeepSeek model with memory-efficient configuration for Apple Silicon."""
    model_name = config.MODEL_NAME

    # Set tokenizer configuration
    tokenizer = get_tokenizer(model_name, 'right')

    device = "mps" if torch.backends.mps.is_available() else "auto"
    # Load model with optimized settings for apple silicon
    model = get_model(model_name, device_map=device, use_cache=True)

    # Apply memory optimizations
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    return model, tokenizer

def setup_trainer(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, train_dataset, eval_dataset):
    """Setup the LoRA training configuration following DeepSeek's recommendations."""
    # LoRA configuration optimized for quick learning
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=4,  # Reduced rank for faster training
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"]
    )
    # Training arguments optimized for speed and Apple Silicon
    training_args = TrainingArguments(
        output_dir=config.OUTPUT_DIR,
        num_train_epochs=1,  # Single epoch for tutorial
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        weight_decay=0.01,
        warmup_ratio=0.03,
        logging_steps=1,
        logging_strategy="steps",
        logging_dir="./logs",
        save_strategy="steps",
        save_steps=5,
        # save_total_limit=1,
        fp16=False,  # Disable mixed precision for Apple Silicon
        bf16=False,
        optim="adamw_torch_fused",  # Use fused optimizer
        report_to="wandb",
        gradient_checkpointing=True,
        group_by_length=True,
        max_grad_norm=0.3,
        dataloader_num_workers=0,
        remove_unused_columns=True,
        run_name=config.OUTPUT_DIR,
        # Memory optimizations
        deepspeed=None,
        local_rank=-1,
        ddp_find_unused_parameters=None,
        torch_compile=False,
        use_mps_device=torch.backends.mps.is_available(),
        disable_tqdm=False,
    )

    # Create data collator for proper padding
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Initialize trainer with processing_class instead of tokenizer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        peft_config=peft_config,
        args=training_args,
        data_collator=data_collator,
        processing_class=None  # Let SFTTrainer handle processing
    )

    return trainer

def prepare_dataset(tokenizer):
    """Prepare the coding dataset for training.

    Following DeepSeek's recommendation to include 'Please reason step by step'
    in the prompt for better reasoning performance.
    """

    if config.preprocess:
        print("Preprocessing dataset...")
        dataset = preprocess.preprocess_dataset()
    else:
        dataset = load_dataset(config.DEFAULT_DATASET, config.DEFAULT_NAME,trust_remote_code=True)
    print(f"Dataset loaded with {len(dataset['train'])} training examples")


    # Tokenize with optimized settings for speed
    train_dataset = dataset['train'].map(
        lambda x: tokenizer(
            config.template.format(*[msg['content'] for msg in x["messages"]]),
            truncation=False,
            padding="max_length",
            max_length=16384,  # Reduced sequence length for faster training (for better results, use 2048)
            return_tensors=None,
        ),
        remove_columns=["reasoning","answer"],
        num_proc=os.cpu_count(),
    )

    print(f"\nUsing {len(train_dataset)} examples for training")
    print("\nSample formatted data:")
    print(train_dataset["train"][0])

    return train_dataset

def test_model(model_path):
    """Test the fine-tuned model following DeepSeek's usage recommendations."""

    test_saved_model(model_path)
#     # Create offload directory if it doesn't exist
#     os.makedirs("offload", exist_ok=True)

#     # Load model with proper memory management for Apple Silicon
#     kwargs = {
#         "offload_folder": "offload",
#         "offload_state_dict": True
#     }
#     model = get_model(model_path, kwargs=kwargs)

#     tokenizer = get_tokenizer(model_path)

#     # Initialize pipeline with optimized settings
#     pipe = get_pipeline(model, tokenizer)

#     # Medical test case with recommended prompt format
#     test_problem = """Please reason step by step:

# A 45-year-old patient presents with sudden onset chest pain, shortness of breath, and anxiety. The pain is described as sharp and worsens with deep breathing. What is the most likely diagnosis and what immediate tests should be ordered?"""

#     try:
#         result = pipe(
#             test_problem,
#             # max_new_tokens=512,
#             # temperature=0.6,
#             # top_p=0.95,
#             # repetition_penalty=1.15
#         )

#         print("\nTest Problem:", test_problem)
#         print("\nModel Response:", result[0]["generated_text"])

#         # Log test results to wandb
#         wandb.log({
#             "test_example": wandb.Table(
#                 columns=["Test Case", "Model Response"],
#                 data=[[test_problem, result[0]["generated_text"]]]
#             )
#         })
#     except Exception as e:
#         print(f"\nError during testing: {str(e)}")
#         print("Model was saved successfully but testing failed. You can load the model separately for testing.")
#     finally:
#         # Clean up
#         if os.path.exists("offload"):
#             import shutil
#             shutil.rmtree("offload")


def main():
    """Main function to run the fine-tuning process."""
    do_wandb_init()
    try:
        print("\nSetting up model...")
        model, tokenizer = setup_model()

        print("\nPreparing dataset...")
        train_dataset = prepare_dataset(tokenizer)

        print("\nSetting up trainer...")
        trainer = setup_trainer(model, tokenizer, train_dataset, None)

        print("\nStarting training...")
        try:
            trainer.train(resume_from_checkpoint=True)
        except ValueError as e:
            print(f"Unable to load from checkpoint: {e}")
            print("Starting training from the beginning...")
            trainer.train()

        print("\nSaving model...")
        trainer.model.save_pretrained("./fine_tuned_model")
        tokenizer.save_pretrained("./fine_tuned_model")

        print("\nTesting model...")
        test_model("./fine_tuned_model")

    finally:
        wandb.finish()

if __name__ == "__main__":
    main()
