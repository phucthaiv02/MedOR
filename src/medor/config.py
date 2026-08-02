from typing import List, Optional

import yaml
from pydantic import BaseModel, Field


def load_yaml_config(path: str, model: type[BaseModel]) -> BaseModel:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return model(**raw)


class LoraConfig(BaseModel):
    r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    target_modules: List[str] = Field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])


class TrainConfig(BaseModel):
    base_model: str = "unsloth/Qwen3-8B"

    train_csv: str = "data/medor/train.csv"
    val_csv: Optional[str] = None
    val_split: float = 0.05
    train_sample_limit: Optional[int] = None

    max_seq_length: int = 4000
    load_in_4bit: bool = False
    seed: int = 42

    lora: LoraConfig = Field(default_factory=LoraConfig)

    output_dir: str = "outputs/qwen3-medor-lora"
    merged_dir: str = "outputs/qwen3-medor-merged"

    num_train_epochs: float = 3.0
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.01
    logging_steps: int = 10
    save_steps: int = 200
    save_total_limit: int = 5
    eval_steps: int = 200
    packing: bool = False

    report_to: str = "wandb"
    wandb_project: Optional[str] = "medor-qwen3"
    wandb_run_name: Optional[str] = None

    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    hub_private: bool = True
    hub_token: Optional[str] = None
    hub_strategy: str = "all_checkpoints"
    hub_merged_branch: str = "merged"


class CandidateKBConfig(BaseModel):
    entity_type: str
    csv_path: str


class SamplingConfig(BaseModel):
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 2048
    repetition_penalty: float = 1.05
    frequency_penalty: float = 0.1
    stop: List[str] = Field(default_factory=lambda: ["<|im_end|>", "<|endoftext|>"])


class InferConfig(BaseModel):
    base_model: str = "unsloth/Qwen3-8B"
    lora_path: Optional[str] = "outputs/qwen3-medor-lora"
    merged_model_path: Optional[str] = "outputs/qwen3-medor-merged"

    input_dir: str = "data/medor/eval/txt"
    output_dir: str = "outputs/predictions"
    limit: Optional[int] = None

    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.85
    tensor_parallel_size: int = 1
    guided_decoding: bool = True

    # Concurrency/throughput tuning for vLLM's scheduler: max_num_seqs bounds how
    # many sequences can be decoded concurrently; max_num_batched_tokens bounds how
    # many tokens (prefill + decode) are packed into one scheduler step. Leave null
    # to use vLLM's defaults.
    max_num_seqs: Optional[int] = None
    max_num_batched_tokens: Optional[int] = None

    sampling: SamplingConfig = Field(default_factory=SamplingConfig)

    # Candidate retrieval: for each entry in `candidate_kbs`, entities whose type
    # matches `entity_type` are matched (via SapBERT embedding similarity) against
    # that entry's `csv_path` (columns: name, code), and the top-k closest codes
    # are attached as a `candidates` field. Disabled when candidate_kbs is empty.
    # e.g. one KB for CHẨN_ĐOÁN (disease codes) and another for THUỐC (drug codes).
    embedding_model: str = "cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR"
    candidate_top_k: int = 1
    candidate_kbs: List[CandidateKBConfig] = Field(default_factory=list)


class EvalConfig(BaseModel):
    predictions_dir: str = "outputs/predictions"
    gold_dir: str = "data/medor/eval/gold"
    metrics_output: str = "outputs/eval/metrics.json"
    match_mode: str = "text_type"
