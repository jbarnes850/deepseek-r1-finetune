from config import config


from datasets import load_dataset           # type: ignore


import os


def prepare_dataset(tokenizer):
    """Prepare the medical reasoning dataset for training.

    Following DeepSeek's recommendation to include 'Please reason step by step'
    in the prompt for better reasoning performance.
    """
    dataset = load_dataset(config.DEFAULT_DATASET, config.DEFAULT_NAME)
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
    print(f"For demo purposes, will use {len(dataset['train'])} training examples now.")

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
            truncation=False,
            padding="max_length",
            max_length=8192,  # Reduced sequence length for faster training (for better results, use 2048)
            return_tensors=None,
        ),
        #remove_columns=["text"],
        num_proc=os.cpu_count()
    )

    print(f"\nUsing {len(train_dataset)} examples for training")
    print("\nSample formatted data:")
    print(format_instruction(dataset["train"][0]))

    return train_dataset
