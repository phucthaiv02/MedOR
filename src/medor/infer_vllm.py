import argparse
import glob
import json
import os

from dotenv import load_dotenv

from .config import InferConfig, load_yaml_config
from .prompts import build_messages
from .schema import entity_list_json_schema


def _sorted_txt_files(input_dir):
    paths = glob.glob(os.path.join(input_dir, "*.txt"))

    def sort_key(path):
        stem = os.path.splitext(os.path.basename(path))[0]
        return (0, int(stem)) if stem.isdigit() else (1, stem)

    return sorted(paths, key=sort_key)


def find_position(input_text, context, text):
    """Locate `text` inside `input_text` via its `context`: first find where
    context sits in input_text, then find text's offset within that context,
    then translate to a global [start, end] offset in input_text."""
    context = (context or "").strip()
    text = (text or "").strip()
    if not text:
        return None

    ctx_start = input_text.find(context) if context else -1
    if ctx_start == -1:
        return None

    local_start = context.find(text)
    if local_start == -1:
        return None

    start = ctx_start + local_start
    return [start, start + len(text)]


def align_entities(input_text, entities):
    aligned = []
    n_dropped = 0
    for ent in entities:
        position = find_position(input_text, ent.get("context", ""), ent.get("text", ""))
        if position is None:
            n_dropped += 1
            continue
        aligned.append({
            "text": ent["text"].strip(),
            "type": ent.get("type", ""),
            "assertions": ent.get("assertions", []),
            "position": position,
        })
    return aligned, n_dropped


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run inference over a directory of .txt files with vLLM")
    parser.add_argument("--config", default="configs/infer.yaml")
    args = parser.parse_args()
    cfg: InferConfig = load_yaml_config(args.config, InferConfig)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from vllm.sampling_params import GuidedDecodingParams

    model_path = cfg.merged_model_path or cfg.base_model
    llm_kwargs = dict(
        model=model_path,
        max_model_len=cfg.max_model_len,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        tensor_parallel_size=cfg.tensor_parallel_size,
    )

    lora_request = None
    if cfg.lora_path and not cfg.merged_model_path:
        llm_kwargs["enable_lora"] = True
        lora_request = LoRARequest("medor-lora", 1, cfg.lora_path)

    llm = LLM(**llm_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    txt_paths = _sorted_txt_files(cfg.input_dir)
    if cfg.limit:
        txt_paths = txt_paths[: cfg.limit]
    if not txt_paths:
        raise FileNotFoundError(f"No .txt files found in {cfg.input_dir}")

    texts = []
    for path in txt_paths:
        with open(path, "r", encoding="utf-8") as f:
            texts.append(f.read())

    # Không dùng system prompt / hướng dẫn: model đã được fine-tune để map
    # thẳng input_text -> JSON, chỉ cần dựng lại đúng chat template lúc train.
    prompts = [
        tokenizer.apply_chat_template(
            build_messages(text), tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        for text in texts
    ]

    guided_decoding = GuidedDecodingParams(json=entity_list_json_schema()) if cfg.guided_decoding else None
    sampling_params = SamplingParams(
        temperature=cfg.sampling.temperature,
        top_p=cfg.sampling.top_p,
        max_tokens=cfg.sampling.max_tokens,
        repetition_penalty=cfg.sampling.repetition_penalty,
        frequency_penalty=cfg.sampling.frequency_penalty,
        stop=cfg.sampling.stop,
        guided_decoding=guided_decoding,
    )

    generate_kwargs = {}
    if lora_request:
        generate_kwargs["lora_request"] = lora_request

    outputs = llm.generate(prompts, sampling_params, **generate_kwargs)

    os.makedirs(cfg.output_dir, exist_ok=True)
    n_invalid = 0
    n_dropped_total = 0
    for path, text, out in zip(txt_paths, texts, outputs):
        stem = os.path.splitext(os.path.basename(path))[0]
        raw_output = out.outputs[0].text.strip()
        try:
            entities = json.loads(raw_output)
            if not isinstance(entities, list):
                raise ValueError("top-level JSON is not a list")
        except (json.JSONDecodeError, ValueError):
            n_invalid += 1
            entities = []

        aligned, n_dropped = align_entities(text, entities)
        n_dropped_total += n_dropped

        out_path = os.path.join(cfg.output_dir, f"{stem}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(aligned, f, ensure_ascii=False, indent=2)

    print(
        f"[INFO] Wrote {len(outputs)} predictions to {cfg.output_dir} "
        f"({n_invalid} invalid JSON outputs, {n_dropped_total} entities dropped due to failed position alignment)"
    )


if __name__ == "__main__":
    main()
