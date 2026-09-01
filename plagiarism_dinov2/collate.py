"""
Figure-Caption Plagiarism Pretraining
Collate (che do image-text)
--------------------------------------------------------------
Tokenize caption_text ngay trong collate_fn de padding dong theo
tung batch (nhanh hon padding co dinh max_length tu dataset).
"""
from __future__ import annotations

import torch


class TextCollator:
    """
    Dung lam collate_fn cho DataLoader khi CAPTION_MODALITY == "text".
    """

    def __init__(self, tokenizer, max_length: int = 128) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: list[dict]) -> dict:
        figures = torch.stack(
            [item["figure"] for item in batch],
            dim=0,
        )

        captions = [item["caption_text"] for item in batch]

        encoded = self.tokenizer(
            captions,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "figure": figures,
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "figure_id": [item["figure_id"] for item in batch],
            "caption_id": [item["caption_id"] for item in batch],
        }
