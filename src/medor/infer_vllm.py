import argparse
import glob
import json
import os
import re

from dotenv import load_dotenv

from .config import InferConfig, load_yaml_config
from .prompts import build_messages
from .schema import entity_list_json_schema

_WHITESPACE_RE = re.compile(r"\s+")


def _sorted_txt_files(input_dir):
    paths = glob.glob(os.path.join(input_dir, "*.txt"))

    def sort_key(path):
        stem = os.path.splitext(os.path.basename(path))[0]
        return (0, int(stem)) if stem.isdigit() else (1, stem)

    return sorted(paths, key=sort_key)


def _normalize_whitespace(text):
    """Collapse every run of whitespace (including newlines) in `text` down
    to a single space, returning the normalized string plus a list mapping
    each of its indices back to the matching index in the original `text`."""
    out = []
    index_map = []
    i, n = 0, len(text)
    while i < n:
        if text[i].isspace():
            out.append(" ")
            index_map.append(i)
            while i < n and text[i].isspace():
                i += 1
        else:
            out.append(text[i])
            index_map.append(i)
            i += 1
    return "".join(out), index_map


def find_position(norm_input, index_map, context, text):
    """Locate `text` inside the original text behind (`norm_input`, `index_map`)
    via its `context`: first find where context sits in norm_input, then find
    text's offset within that context, then translate back to a global
    [start, end] offset in the original text. Matching is whitespace-insensitive
    (any run of spaces/newlines collapses to one space) since the model doesn't
    always reproduce the input's exact newlines verbatim in `context`/`text`."""
    context = (context or "").strip()
    text = (text or "").strip()
    if not text:
        return None

    norm_context = _WHITESPACE_RE.sub(" ", context)
    norm_text = _WHITESPACE_RE.sub(" ", text)

    ctx_start = norm_input.find(norm_context) if norm_context else -1
    if ctx_start == -1:
        return None

    local_start = norm_context.find(norm_text)
    if local_start == -1:
        return None

    norm_start = ctx_start + local_start
    norm_end = norm_start + len(norm_text)
    if norm_end > len(index_map):
        return None

    start = index_map[norm_start]
    end = index_map[norm_end - 1] + 1
    return [start, end]


def align_entities(input_text, entities):
    aligned = []
    n_dropped = 0
    norm_input, index_map = _normalize_whitespace(input_text)
    for ent in entities:
        position = find_position(norm_input, index_map, ent.get("context", ""), ent.get("text", ""))
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


def _write_predictions(txt_paths, all_aligned, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for path, aligned in zip(txt_paths, all_aligned):
        stem = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(output_dir, f"{stem}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(aligned, f, ensure_ascii=False, indent=2)


def _write_raw_predictions(txt_paths, raw_outputs, output_dir):
    """Dump each prompt's unparsed model output verbatim, so a `[]` in the
    aligned predictions can be traced back to what the model actually
    generated (empty array vs. invalid/truncated JSON vs. failed alignment)."""
    raw_dir = os.path.join(output_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    for path, raw in zip(txt_paths, raw_outputs):
        stem = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(raw_dir, f"{stem}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(raw)


def load_embedding_model(embedding_model_name):
    """Load the SapBERT model/tokenizer shared across all candidate KBs.
    Import torch/transformers lazily so the module stays importable without
    them when candidate retrieval is disabled."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
    model = AutoModel.from_pretrained(embedding_model_name).to(device)
    model.eval()
    return tokenizer, model, device


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
    if cfg.max_num_seqs is not None:
        llm_kwargs["max_num_seqs"] = cfg.max_num_seqs
    if cfg.max_num_batched_tokens is not None:
        llm_kwargs["max_num_batched_tokens"] = cfg.max_num_batched_tokens

    lora_request = None
    if cfg.lora_path and not cfg.merged_model_path:
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = cfg.max_lora_rank
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

    n_invalid = 0
    n_dropped_total = 0
    all_aligned = []
    raw_outputs = []
    for text, out in zip(texts, outputs):
        raw_output = out.outputs[0].text.strip()
        raw_outputs.append(raw_output)
        try:
            entities = json.loads(raw_output)
            if not isinstance(entities, list):
                raise ValueError("top-level JSON is not a list")
        except (json.JSONDecodeError, ValueError):
            n_invalid += 1
            entities = []

        aligned, n_dropped = align_entities(text, entities)
        n_dropped_total += n_dropped
        all_aligned.append(aligned)

    _write_predictions(txt_paths, all_aligned, cfg.output_dir)
    _write_raw_predictions(txt_paths, raw_outputs, cfg.output_dir)

    if cfg.candidate_kbs:
        from .candidates import attach_candidates, embed_texts, load_kb

        kb_tokenizer, kb_model, kb_device = load_embedding_model(cfg.embedding_model)
        for kb_cfg in cfg.candidate_kbs:
            kb_names, kb_codes = load_kb(kb_cfg.csv_path)
            kb_embeddings = embed_texts(kb_names, kb_tokenizer, kb_model, kb_device, batch_size=128)
            attach_candidates(
                all_aligned,
                kb_names,
                kb_codes,
                kb_embeddings,
                kb_tokenizer,
                kb_model,
                kb_device,
                top_k=cfg.candidate_top_k,
                target_type=kb_cfg.entity_type,
            )

    _write_predictions(txt_paths, all_aligned, cfg.output_dir)

    print(
        f"[INFO] Wrote {len(outputs)} predictions to {cfg.output_dir} "
        f"({n_invalid} invalid JSON outputs, {n_dropped_total} entities dropped due to failed position alignment)"
    )


if __name__ == "__main__":
    main()
