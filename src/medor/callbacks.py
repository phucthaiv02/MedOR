from transformers import TrainerCallback

from .metrics import (
    assertion_accuracy,
    exact_match,
    match_by_text_type,
    parse_entities,
    precision_recall_f1,
    score,
    word_error_rate,
)
from .prompts import build_messages


class GenerationEvalCallback(TrainerCallback):
    """Runs free-form generation on a slice of the val set at every trainer eval_steps
    and logs entity-level Precision/Recall/F1/Exact Match/Assertion Accuracy plus raw-text
    WER, alongside eval_loss. `trainer` must be set (callback.trainer = trainer) after the
    Trainer is constructed, since Trainer.log() is how these numbers reach wandb/console."""

    def __init__(self, model, tokenizer, val_raw, max_samples, max_new_tokens, match_mode, batch_size=8):
        self.model = model
        self.tokenizer = tokenizer
        n = len(val_raw) if max_samples is None else min(max_samples, len(val_raw))
        self.rows = val_raw.select(range(n))
        self.max_new_tokens = max_new_tokens
        self.match_mode = match_mode
        self.batch_size = batch_size
        self.trainer = None

    def _generate(self, prompts):
        import torch

        self.tokenizer.padding_side = "left"
        texts = []
        for i in range(0, len(prompts), self.batch_size):
            chunk = prompts[i : i + self.batch_size]
            inputs = self.tokenizer(chunk, return_tensors="pt", padding=True).to(self.model.device)
            with torch.no_grad():
                out_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                )
            new_tokens = out_ids[:, inputs["input_ids"].shape[1] :]
            texts.extend(self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
        return [t.strip() for t in texts]

    def on_evaluate(self, args, state, control, **kwargs):
        if self.trainer is None or len(self.rows) == 0:
            return

        from unsloth import FastLanguageModel

        prompts = [
            self.tokenizer.apply_chat_template(
                build_messages(row["input_text"]), tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            for row in self.rows
        ]

        FastLanguageModel.for_inference(self.model)
        try:
            generations = self._generate(prompts)
        finally:
            FastLanguageModel.for_training(self.model)

        total_tp = total_fp = total_fn = 0
        total_exact = total_assertion_correct = total_matched = 0
        total_wer_dist = total_wer_ref_len = 0
        n_invalid_json = 0

        for row, gen_text in zip(self.rows, generations):
            gold_text = row["response"]
            gold_entities = parse_entities(gold_text) or []
            pred_entities = parse_entities(gen_text)
            if pred_entities is None:
                n_invalid_json += 1
                pred_entities = []

            tp, fp, fn = score(gold_entities, pred_entities, self.match_mode)
            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_exact += int(exact_match(gold_entities, pred_entities, self.match_mode))

            correct, matched = assertion_accuracy(match_by_text_type(gold_entities, pred_entities))
            total_assertion_correct += correct
            total_matched += matched

            dist, ref_len = word_error_rate(gold_text, gen_text)
            total_wer_dist += dist
            total_wer_ref_len += ref_len

        n = len(self.rows)
        metrics = precision_recall_f1(total_tp, total_fp, total_fn)
        metrics.update(
            {
                "exact_match": total_exact / n,
                "wer": total_wer_dist / total_wer_ref_len if total_wer_ref_len else 0.0,
                "assertion_accuracy": total_assertion_correct / total_matched if total_matched else 0.0,
                "invalid_json_rate": n_invalid_json / n,
            }
        )
        self.trainer.log({f"eval_{k}": v for k, v in metrics.items()})
