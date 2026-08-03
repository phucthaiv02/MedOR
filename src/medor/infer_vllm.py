import argparse
import glob
import json
import os
import re

from dotenv import load_dotenv

from .config import InferConfig, load_yaml_config
from .prompts import build_messages, build_type_choice_messages
from .schema import MedicalTermType, entity_list_json_schema

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


def find_position(norm_input, index_map, text, occurrence):
    """Locate the `occurrence`-th (0-indexed) match of `text` inside the
    original text behind (`norm_input`, `index_map`), then translate it back
    to a global [start, end] offset in the original text. Matching is
    whitespace-insensitive (any run of spaces/newlines collapses to one
    space) since the model doesn't always reproduce the input's exact
    newlines verbatim in `text`."""
    text = (text or "").strip()
    if not text:
        return None

    norm_text = _WHITESPACE_RE.sub(" ", text)
    if not norm_text:
        return None

    norm_start, search_from = -1, 0
    for _ in range(occurrence + 1):
        norm_start = norm_input.find(norm_text, search_from)
        if norm_start == -1:
            return None
        search_from = norm_start + 1

    norm_end = norm_start + len(norm_text)
    if norm_end > len(index_map):
        return None

    start = index_map[norm_start]
    end = index_map[norm_end - 1] + 1
    return [start, end]


def _extract_json_array_items(raw):
    """Return the substrings of every complete top-level element in a JSON
    array `raw`, even if `raw` is cut off mid-way through its last element
    (e.g. the model hit max_tokens). Tracks bracket depth and quoted strings
    (with escapes) so a `}`/`]`/`,` inside a string value doesn't confuse the
    scan; a dangling partial element at the end is simply left out."""
    if not raw.startswith("["):
        return []
    depth = 0  # bracket depth inside the array; 0 == between elements
    in_string = False
    escape = False
    item_start = None
    items = []
    for i, c in enumerate(raw[1:], start=1):
        if in_string:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_string = False
            continue
        if c == '"':
            in_string = True
            if depth == 0 and item_start is None:
                item_start = i
        elif c in "{[":
            if depth == 0 and item_start is None:
                item_start = i
            depth += 1
        elif c in "}]":
            if depth == 0:
                break  # this is the array's own closing bracket
            depth -= 1
            if depth == 0 and item_start is not None:
                items.append(raw[item_start : i + 1])
                item_start = None
    return items


def parse_entities(raw_output):
    """Parse the model's JSON array of entities, salvaging as many complete
    entities as possible when generation was cut short by max_tokens instead
    of dropping the whole output. Returns (entities, status) where status is
    "ok" (well-formed JSON array), "truncated" (partial output, zero or more
    complete entities recovered), or "invalid" (nothing usable found)."""
    raw_output = raw_output.strip()
    try:
        entities = json.loads(raw_output)
        if isinstance(entities, list):
            return entities, "ok"
    except json.JSONDecodeError:
        pass

    items = _extract_json_array_items(raw_output)
    if not items:
        return [], "invalid"

    entities = []
    for item in items:
        try:
            entities.append(json.loads(item))
        except json.JSONDecodeError:
            continue
    return entities, "truncated"


def align_entities(input_text, entities):
    """Align each entity's `text` to a [start, end] offset in `input_text`.
    Entities are assumed to be emitted in the same left-to-right order they
    appear in the document (true of the current training data), so repeated
    mentions of the same text are matched to successive occurrences in that
    order via a per-text counter, without needing a `context` field."""
    aligned = []
    n_dropped = 0
    norm_input, index_map = _normalize_whitespace(input_text)
    seen_counts = {}
    for ent in entities:
        text = (ent.get("text") or "").strip()
        norm_text = _WHITESPACE_RE.sub(" ", text)
        occurrence = seen_counts.get(norm_text, 0)
        position = find_position(norm_input, index_map, text, occurrence)
        if position is None:
            n_dropped += 1
            continue
        seen_counts[norm_text] = occurrence + 1
        aligned.append({
            "text": text,
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


def _llm_kwargs(cfg, model_path):
    kwargs = dict(
        model=model_path,
        max_model_len=cfg.max_model_len,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        tensor_parallel_size=cfg.tensor_parallel_size,
    )
    if cfg.max_num_seqs is not None:
        kwargs["max_num_seqs"] = cfg.max_num_seqs
    if cfg.max_num_batched_tokens is not None:
        kwargs["max_num_batched_tokens"] = cfg.max_num_batched_tokens
    return kwargs


def _type_snippet(input_text, start, end, window=80):
    """Window of `input_text` around [start, end) with the entity span marked
    by «», giving the type-classification prompt just enough surrounding
    context to disambiguate without needing the whole document."""
    left = input_text[max(0, start - window) : start]
    right = input_text[end : end + window]
    return f"{left}«{input_text[start:end]}»{right}"


def classify_types(cfg, texts, all_aligned):
    """Assign a `type` to every aligned entity using a fresh, non-fine-tuned
    instance of `cfg.base_model` (extraction and typing are separate models:
    the fine-tuned model only extracts `text`/`assertions`), constrained via
    vLLM guided-choice decoding to one of MedicalTermType's values. Mutates
    `all_aligned` in place."""
    refs = []
    snippets = []
    for doc_idx, (text, entities) in enumerate(zip(texts, all_aligned)):
        for ent_idx, ent in enumerate(entities):
            start, end = ent["position"]
            refs.append((doc_idx, ent_idx))
            snippets.append(_type_snippet(text, start, end))

    if not refs:
        return

    import gc

    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    llm = LLM(**_llm_kwargs(cfg, cfg.base_model))

    prompts = [
        tokenizer.apply_chat_template(
            build_type_choice_messages(snippet, all_aligned[doc_idx][ent_idx]["text"]),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for snippet, (doc_idx, ent_idx) in zip(snippets, refs)
    ]
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=16,
        stop=cfg.sampling.stop,
        guided_decoding=GuidedDecodingParams(choice=[t.value for t in MedicalTermType]),
    )
    outputs = llm.generate(prompts, sampling_params)

    for (doc_idx, ent_idx), out in zip(refs, outputs):
        all_aligned[doc_idx][ent_idx]["type"] = out.outputs[0].text.strip()

    del llm
    gc.collect()
    torch.cuda.empty_cache()


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

    import gc

    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from vllm.sampling_params import GuidedDecodingParams

    model_path = cfg.merged_model_path or cfg.base_model
    llm_kwargs = _llm_kwargs(cfg, model_path)

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
    n_truncated = 0
    n_dropped_total = 0
    all_aligned = []
    raw_outputs = []
    for text, out in zip(texts, outputs):
        raw_output = out.outputs[0].text.strip()
        raw_outputs.append(raw_output)
        entities, status = parse_entities(raw_output)
        if status == "invalid":
            n_invalid += 1
        elif status == "truncated":
            n_truncated += 1

        aligned, n_dropped = align_entities(text, entities)
        n_dropped_total += n_dropped
        all_aligned.append(aligned)

    _write_predictions(txt_paths, all_aligned, cfg.output_dir)
    _write_raw_predictions(txt_paths, raw_outputs, cfg.output_dir)

    # Free the extraction engine before loading a fresh one for type classification.
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    classify_types(cfg, texts, all_aligned)
    _write_predictions(txt_paths, all_aligned, cfg.output_dir)

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
        f"({n_invalid} invalid JSON outputs, {n_truncated} truncated outputs with partial entities salvaged, "
        f"{n_dropped_total} entities dropped due to failed position alignment)"
    )


if __name__ == "__main__":
    main()
