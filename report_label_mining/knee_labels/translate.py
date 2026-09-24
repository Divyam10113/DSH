"""Optional machine translation of non-English reports to English (MarianMT / opus-mt).

XLM-R and the char-n-gram model already work across languages, so translation is an *extra*
normalisation step: it lets the English rules fire on every report and gives the English-heavy
patterns more coverage. Run it once in a GPU dev notebook and cache the result as a CSV
(`translations.csv`) in a Kaggle Dataset; later runs just read the cache.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .constants import ID_COL
from .text import split_sentences

DEFAULT_MODEL_TEMPLATE = "Helsinki-NLP/opus-mt-{lang}-en"


def _translate_batch(texts: list[str], model, tokenizer, device, max_len: int) -> list[str]:
    import torch

    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(device)
    with torch.no_grad():
        out = model.generate(**enc, max_length=max_len, num_beams=2)
    return tokenizer.batch_decode(out, skip_special_tokens=True)


def translate_reports(df: pd.DataFrame, langs: pd.Series, text_col: str,
                      model_template: str = DEFAULT_MODEL_TEMPLATE, batch_size: int = 32,
                      max_len: int = 256, cache_path: str | Path | None = None, log=print) -> pd.Series:
    """English text for every row of `df` (English and unsupported-language rows pass through).

    Reports are translated sentence by sentence so long reports never get truncated.
    Returns a Series aligned with `df.index`.
    """
    if cache_path is not None and Path(cache_path).exists():
        cache = pd.read_csv(cache_path).set_index(ID_COL)["text_en"]
        if df[ID_COL].isin(cache.index).all():
            log(f"[translate] using cache {cache_path}")
            return df[ID_COL].map(cache).fillna(df[text_col]).set_axis(df.index)

    import torch
    from transformers import MarianMTModel, MarianTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = df[text_col].copy()
    for lang in sorted(set(langs) - {"en", "unk"}):
        rows = df.index[langs == lang]
        name = model_template.format(lang=lang)
        try:
            tokenizer = MarianTokenizer.from_pretrained(name)
            model = MarianMTModel.from_pretrained(name).to(device).eval()
        except Exception as exc:  # no opus-mt model for this language, or offline without weights
            log(f"[translate] skipping '{lang}' ({len(rows)} reports): {exc.__class__.__name__}")
            continue
        if device.type == "cuda":
            model = model.half()
        log(f"[translate] {lang} -> en: {len(rows)} reports with {name}")

        # Keep original casing/accents for the MT model; split on the same sentence boundaries.
        sentences, owners = [], []
        for idx in rows:
            for s in split_sentences(str(df.at[idx, text_col])):
                sentences.append(s)
                owners.append(idx)
        translated = []
        for i in range(0, len(sentences), batch_size):
            translated += _translate_batch(sentences[i:i + batch_size], model, tokenizer, device, max_len)
        joined = pd.Series(translated, index=owners).groupby(level=0).agg(". ".join)
        result.loc[joined.index] = joined
        del model

    if cache_path is not None:
        pd.DataFrame({ID_COL: df[ID_COL].values, "text_en": result.values}).to_csv(cache_path, index=False)
    return result
