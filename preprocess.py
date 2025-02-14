"""
Dependency of deepseek_finetune, an MPS-optimized finetuning script
***NOTE: Each dataset will likely need its own custom script***

This preprocessing script should be edited for each dataset
that is being loaded and passed into deepseek_finetune. Here's
how you use it:

In config.py, define the following variables within the Config class:
* preprocess = True (triggers this preprocessing script)
* template - a string with {} placeholders, for example (note that
    multi-line strings do not indent each line):
    template = '''<|im_start|>system
{}<|im_end|>
<|im_start|>user
{}<|im_end|>
<|im_start|>assistant
{}<|im_end|>'''

Then, in this preprocess.py script, update preprocess_dataset() so
that it returns a dataset with a messages Feature, which is a list:
[
    {
        "role": "system",
        "content": "system prompt text"
    },
    {
        "role": "user",
        "content": "whatever the user says at the prompt."
    },
    {
        "role": "assistant",
        "content": "the response from the LLM"
    }
]

As long as every entry in the dataset has all three messages, the main
script will substitute the 'content' in the order that each message
appears in the list. No additional sorting or validation is done,
so make sure your list is properly ordered.
"""
import os
import sys
from transformers import (              # type: ignore
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)
from datasets import load_dataset       # type: ignore

from config import config

def preprocess_dataset():
    ds = load_dataset(config.DEFAULT_DATASET, config.DEFAULT_NAME,trust_remote_code=True)

    def add_system_prompt(messages: list[dict[str: str]]):
        # Check if there's a system message in the messages
        has_system = any(msg.get('role') == 'system' for msg in messages)

        # If no system message, prepend the system_prompt
        if not has_system:
            system_msg = {'role': 'system', 'content': config.system_prompt}
            messages.insert(0, system_msg)
        # If there is a system message, prepend the system_prompt to it
        else:
            # Get the existing system message
            existing_system = next(msg for msg in messages if msg.get('role') == 'system')
            # Prepend the system_prompt to its content
            existing_system['content'] = config.system_prompt + "\n" + existing_system['content']

        return messages

    def add_reasoning(messages, reasoning = None, answer = None):
        # If reasoning is provided, add a new message to the end of messages
        msg_content = ''
        if reasoning:
            msg_content += "<think>\n" + reasoning + "</think>"
        if answer:
            if reasoning: msg_content += "\n"
            msg_content += answer
        messages.append({
            'role': 'assistant',
            'content': msg_content
        })
        return messages

    def map_messages(columns_dict):
        messages = columns_dict.get("messages", [{},])
        reasoning = columns_dict.get("reasoning", "")
        answer = columns_dict.get("answer", "")
        return {
            "messages":
                add_reasoning(add_system_prompt(messages), reasoning, answer)
        }

    mapped_ds = ds.map(map_messages,remove_columns=['model'],)

    return mapped_ds
# if __name__=="__main__":
#     preprocess_dataset()
