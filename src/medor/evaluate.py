import argparse
import glob
import json
import os

from .config import EvalConfig, load_yaml_config
from .metrics import (
    assertion_accuracy,
    exact_match,
    match_by_text_type,
    precision_recall_f1,
    score,
)


def load_entities(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else None


def main():
    parser = argparse.ArgumentParser(description="Score predictions/*.json against gold/*.json (matched by filename)")
    parser.add_argument("--config", default="configs/eval.yaml")
    args = parser.parse_args()
    cfg: EvalConfig = load_yaml_config(args.config, EvalConfig)

    gold_paths = sorted(glob.glob(os.path.join(cfg.gold_dir, "*.json")))
    if not gold_paths:
        raise FileNotFoundError(f"No gold .json files found in {cfg.gold_dir}")

    total_tp = total_fp = total_fn = 0
    total_exact = total_assertion_correct = total_matched = 0
    n_examples = n_invalid_json = n_missing_pred = 0

    for gold_path in gold_paths:
        stem = os.path.splitext(os.path.basename(gold_path))[0]
        pred_path = os.path.join(cfg.predictions_dir, f"{stem}.json")
        n_examples += 1

        gold = load_entities(gold_path) or []

        if not os.path.exists(pred_path):
            n_missing_pred += 1
            pred = []
        else:
            pred = load_entities(pred_path)
            if pred is None:
                n_invalid_json += 1
                pred = []

        tp, fp, fn = score(gold, pred, cfg.match_mode)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_exact += int(exact_match(gold, pred, cfg.match_mode))

        correct, matched = assertion_accuracy(match_by_text_type(gold, pred))
        total_assertion_correct += correct
        total_matched += matched

    metrics = {
        "n_examples": n_examples,
        "n_missing_predictions": n_missing_pred,
        "invalid_json_rate": n_invalid_json / n_examples if n_examples else 0.0,
        **precision_recall_f1(total_tp, total_fp, total_fn),
        "exact_match": total_exact / n_examples if n_examples else 0.0,
        "assertion_accuracy": total_assertion_correct / total_matched if total_matched else 0.0,
        "match_mode": cfg.match_mode,
    }

    os.makedirs(os.path.dirname(cfg.metrics_output), exist_ok=True)
    with open(cfg.metrics_output, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"[INFO] Metrics saved to {cfg.metrics_output}")


if __name__ == "__main__":
    main()
