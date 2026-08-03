import json


def parse_entities(text):
    """Parse a JSON entity array. Returns None (not []) when the text is not
    valid JSON or not a list, so callers can tell "invalid" apart from "empty"."""
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    return data if isinstance(data, list) else None


def strip_entity_fields(text, drop_fields=("type", "context")):
    """Re-serialize a JSON entity array with `drop_fields` removed from each
    entity. Used to normalize a raw gold/response string to the shape the
    extraction model is actually trained to produce (text + assertions only,
    since type is now assigned by a separate classification step)."""
    entities = parse_entities(text)
    if entities is None:
        return text
    stripped = [{k: v for k, v in ent.items() if k not in drop_fields} for ent in entities]
    return json.dumps(stripped, ensure_ascii=False)


def entity_key(entity, mode="text_type"):
    text = str(entity.get("text", "")).strip().lower()
    etype = str(entity.get("type", "")).strip()
    if mode == "text_type_assertions":
        return (text, etype, tuple(sorted(entity.get("assertions", []) or [])))
    return (text, etype)


def score(gold_list, pred_list, mode="text_type"):
    """Greedily match pred entities against gold entities by entity_key(mode). Returns (tp, fp, fn)."""
    remaining = [entity_key(e, mode) for e in gold_list]
    tp = 0
    for e in pred_list:
        k = entity_key(e, mode)
        if k in remaining:
            remaining.remove(k)
            tp += 1
    return tp, len(pred_list) - tp, len(remaining)


def precision_recall_f1(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def exact_match(gold_list, pred_list, mode="text_type"):
    """True if the gold and predicted entity sets are identical (order-insensitive)."""
    gold_keys = sorted(entity_key(e, mode) for e in gold_list)
    pred_keys = sorted(entity_key(e, mode) for e in pred_list)
    return gold_keys == pred_keys


def match_by_text_type(gold_list, pred_list):
    """Greedily pair pred entities to gold entities by (text, type) alone, regardless of
    the match_mode used for scoring. Used to check assertion agreement on entities the
    model otherwise got right."""
    remaining = list(gold_list)
    pairs = []
    for pred in pred_list:
        pred_key = entity_key(pred, "text_type")
        for gold in remaining:
            if entity_key(gold, "text_type") == pred_key:
                pairs.append((gold, pred))
                remaining.remove(gold)
                break
    return pairs


def assertion_accuracy(matched_pairs):
    """Among entities matched on (text, type), how many have an identical assertions set.
    Returns (n_correct, n_matched)."""
    correct = 0
    for gold, pred in matched_pairs:
        gold_assertions = tuple(sorted(gold.get("assertions", []) or []))
        pred_assertions = tuple(sorted(pred.get("assertions", []) or []))
        if gold_assertions == pred_assertions:
            correct += 1
    return correct, len(matched_pairs)


def word_error_rate(reference, hypothesis):
    """Word-level edit distance between reference and hypothesis text.
    Returns (distance, n_reference_words) so callers can aggregate distance/ref_words
    across a corpus instead of averaging per-example rates."""
    ref = reference.split()
    hyp = hypothesis.split()
    n, m = len(ref), len(hyp)
    if n == 0:
        return (0 if m == 0 else m), 1  # avoid dividing by zero; matches jiwer's convention
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            temp = dp[j]
            if ref[i - 1] == hyp[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[m], n
