import pandas as pd


def load_kb(csv_path):
    df = pd.read_csv(csv_path)
    names = df["name"].astype(str).tolist()
    codes = df["code"].astype(str).tolist()
    return names, codes


def embed_texts(texts, tokenizer, model, device, batch_size=32, max_length=64):
    import torch
    import torch.nn.functional as F

    if not texts:
        return torch.empty(0, model.config.hidden_size, device=device)

    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            cls_embeds = outputs.last_hidden_state[:, 0, :]
            embeddings.append(F.normalize(cls_embeds, p=2, dim=1))
    return torch.cat(embeddings, dim=0)


def attach_candidates(
    doc_entities,
    kb_names,
    kb_codes,
    kb_embeddings,
    tokenizer,
    model,
    device,
    top_k=1,
    target_type="CHẨN_ĐOÁN",
):
    """Add a `candidates` field (list of KB codes) to every entity whose type
    equals `target_type`, across `doc_entities` (a list of per-document entity
    lists), matched against `kb_names`/`kb_codes` via SapBERT cosine
    similarity. Mutates entities in place."""
    import torch

    refs = []
    queries = []
    for doc_idx, entities in enumerate(doc_entities):
        for ent_idx, ent in enumerate(entities):
            if ent.get("type") == target_type:
                ent.setdefault("candidates", [])
                refs.append((doc_idx, ent_idx))
                queries.append(ent["text"])

    if not queries:
        return

    query_embeddings = embed_texts(queries, tokenizer, model, device)
    k = min(top_k, kb_embeddings.shape[0])
    similarity = torch.mm(query_embeddings, kb_embeddings.T)
    _, top_indices = torch.topk(similarity, k=k, dim=1)

    for row, (doc_idx, ent_idx) in enumerate(refs):
        matched = top_indices[row].tolist()
        doc_entities[doc_idx][ent_idx]["candidates"] = [kb_codes[idx] for idx in matched]
