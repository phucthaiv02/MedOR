import pandas as pd
from datasets import Dataset

from .prompts import build_messages


def load_csv_dataset(path: str) -> Dataset:
    df = pd.read_csv(path)
    return Dataset.from_pandas(df[["input_text", "response"]], preserve_index=False)


def format_for_training(dataset: Dataset, tokenizer, max_seq_length: int) -> Dataset:
    """Render each example as a chat-formatted string and drop samples whose
    token length is not strictly below max_seq_length."""

    def _to_text(example):
        messages = build_messages(example["input_text"], example["response"])
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
        )
        return {"text": text}

    dataset = dataset.map(_to_text, remove_columns=dataset.column_names)

    def _under_max_length(example):
        n_tokens = len(tokenizer(example["text"], add_special_tokens=False)["input_ids"])
        return n_tokens < max_seq_length

    before = len(dataset)
    dataset = dataset.filter(_under_max_length)
    dropped = before - len(dataset)
    if dropped:
        print(f"[INFO] Dropped {dropped}/{before} samples with context length >= {max_seq_length} tokens")
    return dataset
