import pandas as pd
from datasets import Dataset

from .prompts import build_prompt_completion


def load_csv_dataset(path: str) -> Dataset:
    df = pd.read_csv(path)
    return Dataset.from_pandas(df[["input_text", "response"]], preserve_index=False)


def format_for_training(dataset: Dataset, tokenizer, max_seq_length: int) -> Dataset:
    """Render each example as a TRL prompt/completion message pair (system+user vs.
    assistant JSON) so SFTTrainer's completion_only_loss computes the training loss
    only on the JSON response, not the system instruction or input document. Drops
    samples whose full (prompt+completion) token length is not strictly below
    max_seq_length."""

    def _to_prompt_completion(example):
        prompt, completion = build_prompt_completion(example["input_text"], example["response"])
        return {"prompt": prompt, "completion": completion, "chat_template_kwargs": {"enable_thinking": False}}

    dataset = dataset.map(_to_prompt_completion, remove_columns=dataset.column_names)

    def _under_max_length(example):
        full_ids = tokenizer.apply_chat_template(
            example["prompt"] + example["completion"], tokenize=True, enable_thinking=False
        )
        return len(full_ids) < max_seq_length

    before = len(dataset)
    dataset = dataset.filter(_under_max_length)
    dropped = before - len(dataset)
    if dropped:
        print(f"[INFO] Dropped {dropped}/{before} samples with context length >= {max_seq_length} tokens")
    return dataset
