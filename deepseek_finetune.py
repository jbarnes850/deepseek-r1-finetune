"""
Fine-tuning script for DeepSeek-R1-Distill-Llama-8B model on Apple Silicon.
This tutorial demonstrates how to fine-tune the model for medical reasoning tasks.
Optimized for M2/M3 Macs with ~36GB RAM.

References:
- Model: https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B
- Dataset: https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT
"""

import os
import torch
import platform
import wandb
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    pipeline,
    logging
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

# Initialize wandb for experiment tracking
wandb.init(
    project="deepseek-r1-medical-finetuning",
    config={
        "model_name": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "learning_rate": 5e-5,
        "batch_size": 1,
        "num_epochs": 2,
        "hardware": "Apple Silicon",
        "dataset": "medical-o1-reasoning-SFT",
        "lora_rank": 8,
        "lora_alpha": 16
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

def prepare_dataset(tokenizer):
    """Prepare the medical reasoning dataset for training.
    
    Following DeepSeek's recommendation to include 'Please reason step by step'
    in the prompt for better reasoning performance.
    """
    dataset = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", "en")
    print(f"Dataset loaded with {len(dataset['train'])} training examples")
    
    def format_instruction(sample):
        # Following DeepSeek's recommended prompt format
        return f"""Please reason step by step:

Question: {sample['Question']}

Let's solve this step by step:
{sample['Complex_CoT']}

Final Answer: {sample['Response']}"""
    
    # Use 5% of data for faster tutorial run
    dataset = dataset["train"].train_test_split(train_size=0.05, test_size=0.01, seed=42)
    
    # Prepare training dataset
    train_dataset = dataset["train"].map(
        lambda x: {"text": format_instruction(x)},
        remove_columns=dataset["train"].column_names,
        num_proc=os.cpu_count()
    )
    
    # Tokenize with optimized settings for speed
    train_dataset = train_dataset.map(
        lambda x: tokenizer(
            x["text"],
            truncation=True,
            padding="max_length",
            max_length=1024,  # Reduced sequence length for faster training
            return_tensors=None,
        ),
        remove_columns=["text"],
        num_proc=os.cpu_count()
    )
    
    print(f"\nUsing {len(train_dataset)} examples for training")
    print("\nSample formatted data:")
    print(format_instruction(dataset["train"][0]))
    
    return train_dataset

def setup_model():
    """Setup the DeepSeek model with memory-efficient configuration for Apple Silicon."""
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    
    # Set tokenizer configuration first
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Set padding side before model loading
    
    # Load model with optimized settings for M2/M3
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="mps" if torch.backends.mps.is_available() else "auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        use_cache=False,  # Disable KV-cache to save memory
        max_memory={0: "24GB"},  # Reserve memory for training
    )
    
    # Apply memory optimizations
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    
    return model, tokenizer

def setup_trainer(model, tokenizer, train_dataset, eval_dataset):
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
    
    # Training arguments optimized for speed
    training_args = TrainingArguments(
        output_dir="deepseek-r1-medical-finetuning",
        num_train_epochs=1,  # Single epoch for tutorial
        per_device_train_batch_size=2,  # Increased batch size
        gradient_accumulation_steps=4,  # Reduced steps
        learning_rate=1e-4,  # Increased for faster convergence
        weight_decay=0.01,
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        fp16=False,
        bf16=False,
        optim="adamw_torch",
        report_to="wandb",
        gradient_checkpointing=True,
        group_by_length=True,
        max_grad_norm=0.3,
        dataloader_num_workers=0,
        remove_unused_columns=True,
        run_name="deepseek-medical-tutorial",
        # Memory optimizations
        deepspeed=None,
        local_rank=-1,
        ddp_find_unused_parameters=False,
        torch_compile=False,
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        peft_config=peft_config,
        args=training_args,
        tokenizer=tokenizer,
    )
    
    return trainer

def test_model(model_path):
    """Test the fine-tuned model following DeepSeek's usage recommendations."""
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto"
    )
    
    # Medical test case with recommended prompt format
    test_problem = """Please reason step by step:

A 45-year-old patient presents with sudden onset chest pain, shortness of breath, and anxiety. The pain is described as sharp and worsens with deep breathing. What is the most likely diagnosis and what immediate tests should be ordered?"""
    
    result = pipe(
        test_problem,
        max_new_tokens=512,
        temperature=0.6,  # DeepSeek recommended temperature
        top_p=0.95,
        repetition_penalty=1.15
    )
    
    print("\nTest Problem:", test_problem)
    print("\nModel Response:", result[0]["generated_text"])
    
    # Log test results to wandb
    wandb.log({
        "test_example": wandb.Table(
            columns=["Test Case", "Model Response"],
            data=[[test_problem, result[0]["generated_text"]]]
        )
    })

def main():
    """Main function to run the fine-tuning process."""
    try:
        print("\nSetting up model...")
        model, tokenizer = setup_model()
        
        print("\nPreparing dataset...")
        train_dataset = prepare_dataset(tokenizer)
        
        print("\nSetting up trainer...")
        trainer = setup_trainer(model, tokenizer, train_dataset, None)
        
        print("\nStarting training...")
        trainer.train()
        
        print("\nSaving model...")
        trainer.model.save_pretrained("./fine_tuned_model")
        
        print("\nTesting model...")
        test_model("./fine_tuned_model")
        
    finally:
        wandb.finish()

if __name__ == "__main__":
    main() 