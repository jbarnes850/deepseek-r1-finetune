import token
from transformers import (      # type: ignore
    AutoTokenizer,
    AutoModelForCausalLM,
    Pipeline,
    pipeline
)
from datasets import Dataset, DatasetDict       # type: ignore
from typing import Literal
import torch
from config import config

def get_tokenizer(model_name: str, padding_side: Literal['left']|Literal['right'] = "right", *args, **kwargs) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return tokenizer

def get_model(model_name: str, device_map: str = "auto", use_cache: bool = False, max_memory: str|dict = "27GB", *args, **kwargs) -> AutoModelForCausalLM:
    if isinstance(max_memory, str):
        max_memory = "".join(max_memory.strip().split(" ")).upper()     # allow for sloppy input
        max_memory = {0: max_memory}
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name,
        device_map=device_map,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        use_cache=use_cache,  # Disable KV-cache to save memory
        max_memory=max_memory,  # Reserve memory for training
        *args, **kwargs
    )

    return model


def get_pipeline(model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        max_new_tokens=8192,
        temperature=0.3,      # DeepSeek recommended temperature
        top_p=0.95,
        repetition_penalty=1.15,
        pad_token_id=tokenizer.eos_token_id
    )
    return pipe
