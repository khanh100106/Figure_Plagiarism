"""
Paper2Fig-2026 Retrieval Framework

Retrieval Evaluator
-------------------

Responsibilities
----------------
- Compute retrieval evaluation metrics.
- Keep evaluation logic independent from Trainer.
- Support future retrieval metrics.

Author
------
Nguyen Khanh
"""

from __future__ import annotations

from typing import Dict

import torch

import torch.nn.functional as F


# ============================================================
# Retrieval Evaluator
# ============================================================

class RetrievalEvaluator:
    """
    Retrieval evaluation helper.

    Responsibilities
    ----------------
    - Aggregate embeddings.
    - Compute retrieval metrics.
    - Return metrics as a dictionary.

    Notes
    -----
    Trainer should not know how retrieval metrics
    are computed. All evaluation logic belongs here.
    """

    def evaluate(
            self,
            query_embeddings: torch.Tensor,
            target_embeddings: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Evaluate retrieval performance.
        """

        query_embeddings = F.normalize(
            query_embeddings,
            p=2,
            dim=1,
        )

        target_embeddings = F.normalize(
            target_embeddings,
            p=2,
            dim=1,
        )

        return {
            "recall1": self.compute_recall_at_k(
                query_embeddings,
                target_embeddings,
                k=1,
            ),
            "recall5": self.compute_recall_at_k(
                query_embeddings,
                target_embeddings,
                k=5,
            ),
        }

    @staticmethod
    # ============================================================
    # Cosine Similarity
    # ============================================================
    def compute_cosine_similarity(
            embeddings1: torch.Tensor,
            embeddings2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute cosine similarity.
        Parameters
        ----------
        embeddings1 : torch.Tensor
        embeddings2 : torch.Tensor
        Returns
        -------
        torch.Tensor
        """
        embeddings1 = F.normalize(
            embeddings1,
            p=2,
            dim=1,
        )
        embeddings2 = F.normalize(
            embeddings2,
            p=2,
            dim=1,
        )
        similarity = torch.sum(
            embeddings1 * embeddings2,
            dim=1,
        )
        return similarity

    # ============================================================
    # Embedding Distance
    # ============================================================
    @staticmethod
    def compute_embedding_distance(
            embeddings1: torch.Tensor,
            embeddings2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Euclidean distance.
        """
        return torch.norm(
            embeddings1 - embeddings2,
            dim=1,
        )

    # ============================================================
    # Recall@K
    # ============================================================
    @staticmethod
    def compute_recall_at_k(
            query_embeddings: torch.Tensor,
            target_embeddings: torch.Tensor,
            k: int = 1,
    ) -> float:
        """
        Compute Recall@K.
        Parameters
        ----------
        query_embeddings : torch.Tensor
        target_embeddings : torch.Tensor
        k : int
        Returns
        -------
        float
        """
        query_embeddings = F.normalize(
            query_embeddings,
            p=2,
            dim=1,
        )
        target_embeddings = F.normalize(
            target_embeddings,
            p=2,
            dim=1,
        )
        similarity = torch.matmul(
            query_embeddings,
            target_embeddings.T,
        )
        indices = torch.topk(
            similarity,
            k=k,
            dim=1,
        ).indices
        ground_truth = torch.arange(
            query_embeddings.size(0),
            device=query_embeddings.device,
        ).unsqueeze(1)
        correct = (
                indices == ground_truth
        ).any(
            dim=1,
        )
        recall = correct.float().mean()
        return recall.item()