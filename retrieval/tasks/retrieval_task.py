"""
Paper2Fig-2026
Retrieval Task
"""

from __future__ import annotations

import torch

from retrieval.tasks.base_task import BaseTask

from retrieval.training.metrics import (
    compute_recall_at_k,
)


class RetrievalTask(BaseTask):
    """
    Retrieval task.

    Encapsulates

    - forward
    - loss
    - retrieval metrics
    """

    def forward(
        self,
        batch: dict,
    ) -> dict:

        outputs = {}

        if "pair" in batch:

            pair = batch["pair"]

            outputs["pair"] = {

                "anchor": self.model(
                    pair["anchor"],
                ),

                "target": self.model(
                    pair["target"],
                ),

                "label": pair["label"],

            }

        if "triplet" in batch:

            triplet = batch["triplet"]

            outputs["triplet"] = {

                "anchor": self.model(
                    triplet["anchor"],
                ),

                "positive": self.model(
                    triplet["positive"],
                ),

                "negative": self.model(
                    triplet["negative"],
                ),

            }

        return outputs

    def compute_loss(
        self,
        outputs: dict,
    ) -> dict:

        pair = outputs.get("pair")

        triplet = outputs.get("triplet")

        return self.criterion(

            pair_anchor=(
                pair["anchor"]
                if pair is not None
                else None
            ),

            pair_target=(
                pair["target"]
                if pair is not None
                else None
            ),

            pair_label=(
                pair["label"]
                if pair is not None
                else None
            ),

            triplet_anchor=(
                triplet["anchor"]
                if triplet is not None
                else None
            ),

            triplet_positive=(
                triplet["positive"]
                if triplet is not None
                else None
            ),

            triplet_negative=(
                triplet["negative"]
                if triplet is not None
                else None
            ),

        )

    @torch.no_grad()
    def compute_metrics(
        self,
        outputs: dict,
    ) -> dict:

        metrics = {}

        if "pair" in outputs:

            recall1 = compute_recall_at_k(

                outputs["pair"]["anchor"],

                outputs["pair"]["target"],

                k=1,

            )

            recall5 = compute_recall_at_k(

                outputs["pair"]["anchor"],

                outputs["pair"]["target"],

                k=5,

            )

            metrics["recall1"] = recall1

            metrics["recall5"] = recall5

        return metrics