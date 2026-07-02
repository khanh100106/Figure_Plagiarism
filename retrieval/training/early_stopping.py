"""
Paper2Fig-2026 Retrieval Training
Early Stopping
"""
from __future__ import annotations
from retrieval.constants import (
    DIRECTION_MAX,
    DIRECTION_MIN,
    EARLY_STOP_REASON_PATIENCE,
    EARLY_STOP_REASON_THRESHOLD,
)
class EarlyStopping:
    def __init__(
            self,
            *,
            enabled: bool = True,
            monitor: str = "recall1",
            direction: str = "max",
            patience: int = 10,
            delta: float = 1e-4,
            threshold: float | None = None,
    ) -> None:
        self.enabled = enabled
        self.monitor = monitor
        self.direction = direction
        self.patience = patience
        self.delta = delta
        self.threshold = threshold
        self.best_score = None
        self.best_epoch = 0
        self.counter = 0
        self.should_stop = False
    def reset(
            self,
    ) -> None:
        self.best_score = None
        self.best_epoch = 0
        self.counter = 0
        self.should_stop = False
    def state_dict(
            self,
    ) -> dict:
        return {
            "enabled": self.enabled,
            "monitor": self.monitor,
            "direction": self.direction,
            "patience": self.patience,
            "delta": self.delta,
            "threshold": self.threshold,
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "counter": self.counter,
            "should_stop": self.should_stop,
        }
    def load_state_dict(
            self,
            state: dict,
    ) -> None:
        self.enabled = state["enabled"]
        self.monitor = state["monitor"]
        self.direction = state["direction"]
        self.patience = state["patience"]
        self.delta = state["delta"]
        self.threshold = state["threshold"]
        self.best_score = state["best_score"]
        self.best_epoch = state["best_epoch"]
        self.counter = state["counter"]
        self.should_stop = state["should_stop"]
    def _is_improved(
            self,
            score: float,
    ) -> bool:
        """
        Check whether current score is improved.
        """
        if self.best_score is None:
            return True
        if self.direction == DIRECTION_MAX:
            return (
                    score >
                    self.best_score + self.delta
            )
        if self.direction == DIRECTION_MIN:
            return (
                    score <
                    self.best_score - self.delta
            )
        raise ValueError(
            f"Unknown direction: {self.direction}"
        )
    def step(
            self,
            *,
            epoch: int,
            metrics: dict,
    ) -> dict:
        """
        Update early stopping status.
        Parameters
        ----------
        epoch : int
        metrics : dict
        Returns
        -------
        dict
        """
        if not self.enabled:
            return {
                "stop": False,
                "improved": False,
                "reason": None,
                "best_score": self.best_score,
                "best_epoch": self.best_epoch,
                "counter": self.counter,
            }
        score = metrics[self.monitor]
        improved = self._is_improved(
            score,
        )
        reason = None
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
        #
        # Threshold
        #
        if self.threshold is not None:
            if (
                    self.direction == DIRECTION_MAX
                    and
                    score >= self.threshold
            ):
                self.should_stop = True
                reason = EARLY_STOP_REASON_THRESHOLD
            elif (
                    self.direction == DIRECTION_MIN
                    and
                    score <= self.threshold
            ):
                self.should_stop = True
                reason = EARLY_STOP_REASON_THRESHOLD
        #
        # Patience
        #
        if (
                not self.should_stop
                and
                self.counter >= self.patience
        ):
            self.should_stop = True
            reason = EARLY_STOP_REASON_PATIENCE
        return {
            "stop": self.should_stop,
            "improved": improved,
            "reason": reason,
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "counter": self.counter,
        }
