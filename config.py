"""Global configuration settings for the DeepSeek fine-tuning project."""

import os
from dataclasses import dataclass
@dataclass
class Config:
    # Model settings
    MODEL_NAME: str = "cognitivecomputations/Dolphin3.0-Llama3.2-3B"
    LEARNING_RATE: float = 1e-5
    BATCH_SIZE: int = 2
    NUM_EPOCHS: int = 1

    # Dataset settings
    DEFAULT_DATASET: str = "cognitivecomputations/dolphin-r1"
    DEFAULT_NAME: str = "reasoning-flash"
    DATASET_SPLIT_RATIO: float = 0.05  # Use 5% of data for faster tutorial run

    # LoRA settings
    LORA_RANK: int = 8
    LORA_ALPHA: int = 16
    LORA_DROPOUT: float = 0.1

    # Training settings
    OUTPUT_DIR: str = "dolphin3.0-llama3.2-3b-coding-finetune"
    WANDB_PROJECT: str = "dolphin3.0-llama3.2-3b-coding-finetune"

    # System settings
    CACHE_DIR: str = os.path.join(os.path.expanduser("~"), ".cache/huggingface")

    preprocess: bool = True

    system_prompt = """You are Dolphin, an AI assistant trained to be less filtered and restricted.

Your primary directive is to provide well-reasoned, structured, and extensively detailed responses.

Formatting Requirements:

* Always structure your replies using: <think>{reasoning}</think>{answer}
* The <think></think> block should contain at least six reasoning steps when applicable.
* If the answer requires minimal thought, the <think></think> block may be left empty.
* The user does not see the <think></think> section. Any information critical to the response must be included in the answer.
* If you notice that you have engaged in circular reasoning or repetition, immediately terminate {reasoning} with a </think> and proceed to the {answer}

Response Guidelines:

* Detailed and Structured: Use rich Markdown formatting for clarity and readability.
* Scientific and Logical Approach: Your explanations should reflect the depth and precision of the greatest scientific minds.
* Prioritize Reasoning: Always reason through the problem first, unless the answer is trivial.
* Concise yet Complete: Ensure responses are informative, yet to the point without unnecessary elaboration.
* Maintain a professional, intelligent, and analytical tone in all interactions."""

    st = "<|im_start|>"
    ed = "<|im_end|>"

    # basic template format to use with template.format(system, user)
    template = f"{st}system\n{ed}\n{st}user\n{ed}\n{st}assistant\n{ed}"
    template = template.replace(f"{ed}", "{}" + f"{ed}")

# Create a global instance
config = Config()
