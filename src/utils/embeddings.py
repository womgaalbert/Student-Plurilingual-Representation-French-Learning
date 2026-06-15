"""
embeddings.py — CamemBERT feature extraction + PCA for H3 / H4
French-Learning-Perceptions ML

Usage (called from preprocess.py):
    from utils.embeddings import build_camembert_features
    emb_df = build_camembert_features(df, text_cols, n_components=30)
"""
import logging

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

log = logging.getLogger(__name__)

# Lazy imports — torch & transformers are heavy (only needed for CamemBERT)
_TORCH = None
_TRANSFORMERS_AVAILABLE = False


def _ensure_torch_and_transformers():
    """Lazy-load torch and transformers (only when CamemBERT is actually used)."""
    global _TORCH
    if _TORCH is not None:
        return _TORCH
    import torch as _t
    from transformers import AutoTokenizer, AutoModel  # noqa: F811
    _TORCH = _t
    return _t


def _mean_pool(
    texts: list[str],
    tokenizer,
    model,
    batch_size: int = 16,
    max_length: int = 128,
    device: str = "cpu",
) -> np.ndarray:
    """Mean-pooled CamemBERT embeddings (shape: N × 768)."""
    torch_mod = _ensure_torch_and_transformers()
    all_embs = []
    model.eval()
    with torch_mod.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = [str(t) if pd.notna(t) else "" for t in texts[i : i + batch_size]]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            all_embs.append(emb.cpu().numpy())
    return np.vstack(all_embs)


def build_camembert_features(
    df: pd.DataFrame,
    text_cols: list[str],
    n_components: int = 30,
    model_name: str = "camembert-base",
    device: str = "cpu",
    batch_size: int = 16,
    max_length: int = 128,
    prefix: str = "emb_pca",
) -> pd.DataFrame:
    """
    Compute CamemBERT embeddings on `text_cols`, reduce to `n_components`
    via PCA, and return a DataFrame with columns `{prefix}_0 … {prefix}_{n-1}`
    aligned to df.index.

    Text columns are concatenated row-wise before encoding so the model
    captures cross-field semantics (e.g. attitude + motivation together).
    """
    present = [c for c in text_cols if c in df.columns]
    if not present:
        log.warning("build_camembert_features: aucune colonne texte trouvée dans df.")
        return pd.DataFrame(index=df.index)

    _ensure_torch_and_transformers()
    from transformers import AutoTokenizer, AutoModel
    log.info(f"CamemBERT — chargement du modèle ({model_name}) sur {device}…")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)

    # Concaténer les colonnes texte en une seule phrase par ligne
    corpus = (
        df[present]
        .fillna("")
        .apply(lambda row: " | ".join(v for v in row.values if str(v).strip()), axis=1)
        .tolist()
    )

    log.info(f"CamemBERT — encodage de {len(corpus)} textes (batch={batch_size})…")
    embs = _mean_pool(corpus, tokenizer, model, batch_size, max_length, device)

    n_comp = min(n_components, embs.shape[0] - 1, embs.shape[1])
    log.info(f"CamemBERT — PCA {embs.shape[1]} → {n_comp} composantes…")
    pca = PCA(n_components=n_comp, random_state=42)
    reduced = pca.fit_transform(embs)
    explained = pca.explained_variance_ratio_.sum()
    log.info(f"CamemBERT — variance expliquée par PCA({n_comp}): {explained:.1%}")

    cols = [f"{prefix}_{i}" for i in range(n_comp)]
    return pd.DataFrame(reduced, columns=cols, index=df.index)
