"""Multilingual transformer classifier (XLM-RoBERTa by default) with 12 sigmoid heads.

Runs on a Kaggle GPU. `model_name` can be a Hub id (dev notebooks have internet) or a local path to
weights attached as a Kaggle Dataset (offline). Torch/transformers are imported lazily so the rest
of the package works without them.

Details that matter for this data:
  * masked BCE — partially labeled studies only contribute loss for the labels they have;
  * per-label pos_weight — rare findings are not drowned out (macro-AUC weights them equally);
  * head+tail truncation — reports put the impression/conclusion at the end, so for long reports
    we keep the first `head_tokens` and the last (max_len - head_tokens) tokens.
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import LABELS


@dataclass
class XLMRConfig:
    model_name: str = "xlm-roberta-base"
    max_len: int = 512
    head_tokens: int = 320
    batch_size: int = 16
    grad_accum: int = 1
    lr: float = 2e-5
    head_lr: float = 1e-3
    weight_decay: float = 0.01
    epochs: int = 4
    warmup_frac: float = 0.1
    max_pos_weight: float = 20.0
    fp16: bool = True
    seed: int = 42
    num_workers: int = 0  # inputs are pre-tokenised; collation is cheap


def _seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class XLMRLabeler:
    def __init__(self, config: XLMRConfig | None = None):
        self.config = config or XLMRConfig()
        self.tokenizer = None
        self.model = None

    # -- encoding --------------------------------------------------------------------------------

    def _encode(self, texts: pd.Series) -> list[list[int]]:
        cfg, tok = self.config, self.tokenizer
        body = cfg.max_len - 2  # room for <s> and </s>
        head = min(cfg.head_tokens, body)
        ids = []
        for text in texts:
            t = tok(str(text), add_special_tokens=False, truncation=False)["input_ids"]
            if len(t) > body:
                t = t[:head] + t[len(t) - (body - head):]
            ids.append([tok.cls_token_id, *t, tok.sep_token_id])
        return ids

    def _loader(self, ids: list[list[int]], Y: np.ndarray | None, shuffle: bool):
        import torch
        from torch.utils.data import DataLoader

        pad = self.tokenizer.pad_token_id

        def collate(batch):
            idx = [b[0] for b in batch]
            longest = max(len(ids[i]) for i in idx)
            input_ids = torch.full((len(idx), longest), pad, dtype=torch.long)
            attention = torch.zeros((len(idx), longest), dtype=torch.long)
            for r, i in enumerate(idx):
                input_ids[r, :len(ids[i])] = torch.tensor(ids[i])
                attention[r, :len(ids[i])] = 1
            out = {"input_ids": input_ids, "attention_mask": attention}
            if Y is not None:
                out["labels"] = torch.tensor(Y[idx], dtype=torch.float32)
            return out

        # Sort by length at inference to minimise padding; shuffle for training.
        order = list(range(len(ids)))
        if not shuffle:
            order.sort(key=lambda i: len(ids[i]))
        dataset = [(i,) for i in order]
        return DataLoader(dataset, batch_size=self.config.batch_size, shuffle=shuffle, collate_fn=collate,
                          num_workers=self.config.num_workers), order

    def _load_backbone(self, source: str):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(source)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            source, num_labels=len(LABELS), problem_type="multi_label_classification",
            ignore_mismatched_sizes=True,
        )

    # -- training --------------------------------------------------------------------------------

    def fit(self, texts: pd.Series, Y: np.ndarray, log=print) -> "XLMRLabeler":
        """Fine-tune on reports with a (n, 12) label matrix; NaN cells are masked out of the loss."""
        import torch
        from transformers import get_linear_schedule_with_warmup

        cfg = self.config
        _seed_everything(cfg.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_backbone(cfg.model_name)
        self.model.to(device)

        ids = self._encode(texts)
        loader, _ = self._loader(ids, Y, shuffle=True)

        pos = np.nansum(Y, axis=0)
        neg = np.sum(~np.isnan(Y), axis=0) - pos
        pos_weight = torch.tensor(np.clip(neg / np.maximum(pos, 1), 1.0, cfg.max_pos_weight),
                                  dtype=torch.float32, device=device)

        head_params = [p for n, p in self.model.named_parameters() if "classifier" in n]
        body_params = [p for n, p in self.model.named_parameters() if "classifier" not in n]
        optimizer = torch.optim.AdamW([
            {"params": body_params, "lr": cfg.lr},
            {"params": head_params, "lr": cfg.head_lr},
        ], weight_decay=cfg.weight_decay)
        steps = math.ceil(len(loader) / cfg.grad_accum) * cfg.epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, int(cfg.warmup_frac * steps), steps)
        use_amp = cfg.fp16 and device.type == "cuda"
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        else:  # torch < 2.3
            scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        self.model.train()
        for epoch in range(cfg.epochs):
            total, n_batches = 0.0, 0
            for step, batch in enumerate(loader):
                batch = {k: v.to(device) for k, v in batch.items()}
                labels = batch.pop("labels")
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    logits = self.model(**batch).logits
                mask = ~torch.isnan(labels)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits.float(), torch.nan_to_num(labels), pos_weight=pos_weight, reduction="none")
                loss = (loss * mask).sum() / mask.sum().clamp(min=1)
                scaler.scale(loss / cfg.grad_accum).backward()
                if (step + 1) % cfg.grad_accum == 0 or step + 1 == len(loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                total += loss.item()
                n_batches += 1
            log(f"[xlmr] epoch {epoch + 1}/{cfg.epochs} loss={total / max(n_batches, 1):.4f}")
        return self

    # -- inference -------------------------------------------------------------------------------

    def predict_proba(self, texts: pd.Series) -> np.ndarray:
        import torch

        device = next(self.model.parameters()).device
        ids = self._encode(texts)
        loader, order = self._loader(ids, None, shuffle=False)
        self.model.eval()
        chunks = []
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.autocast(device_type=device.type, enabled=self.config.fp16 and device.type == "cuda"):
                    chunks.append(torch.sigmoid(self.model(**batch).logits.float()).cpu().numpy())
        P_sorted = np.concatenate(chunks) if chunks else np.zeros((0, len(LABELS)))
        P = np.empty_like(P_sorted)
        P[order] = P_sorted
        return P

    # -- persistence -----------------------------------------------------------------------------

    def save(self, out_dir: str | Path) -> None:
        import json

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(out_dir)
        self.tokenizer.save_pretrained(out_dir)
        (out_dir / "labeler_config.json").write_text(json.dumps(asdict(self.config), indent=2))

    @classmethod
    def load(cls, model_dir: str | Path) -> "XLMRLabeler":
        import json

        import torch

        model_dir = Path(model_dir)
        config = XLMRConfig(**json.loads((model_dir / "labeler_config.json").read_text()))
        labeler = cls(config)
        labeler._load_backbone(str(model_dir))
        labeler.model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        return labeler
