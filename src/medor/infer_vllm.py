import argparse
import glob
import json
import os

from dotenv import load_dotenv

from .config import InferConfig, load_yaml_config
from .prompts import build_messages


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run inference over a directory of .txt files with vLLM")
    parser.add_argument("--config", default="configs/infer.yaml")
    args = parser.parse_args()
    cfg: InferConfig = load_yaml_config(args.config, InferConfig)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

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

    txt_paths = sorted(glob.glob(os.path.join(cfg.input_dir, "*.txt")))
    if cfg.limit:
        txt_paths = txt_paths[: cfg.limit]
    if not txt_paths:
        raise FileNotFoundError(f"No .txt files found in {cfg.input_dir}")

    texts = []
    for path in txt_paths:
        with open(path, "r", encoding="utf-8") as f:
            texts.append(f.read())

    prompts = [
        tokenizer.apply_chat_template(build_messages(text), tokenize=False, add_generation_prompt=True)
        for text in texts
    ]

    sampling_params = SamplingParams(
        temperature=cfg.sampling.temperature,
        top_p=cfg.sampling.top_p,
        max_tokens=cfg.sampling.max_tokens,
    )

    generate_kwargs = {}
    if lora_request:
        generate_kwargs["lora_request"] = lora_request

    outputs = llm.generate(prompts, sampling_params, **generate_kwargs)

    os.makedirs(cfg.output_dir, exist_ok=True)
    n_invalid = 0
    for path, out in zip(txt_paths, outputs):
        stem = os.path.splitext(os.path.basename(path))[0]
        raw_output = out.outputs[0].text.strip()
        try:
            entities = json.loads(raw_output)
            if not isinstance(entities, list):
                raise ValueError("top-level JSON is not a list")
            payload = entities
        except (json.JSONDecodeError, ValueError):
            n_invalid += 1
            payload = {"error": "invalid_json", "raw_output": raw_output}

        out_path = os.path.join(cfg.output_dir, f"{stem}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Wrote {len(outputs)} predictions to {cfg.output_dir} ({n_invalid} invalid JSON outputs)")


if __name__ == "__main__":
    main()
