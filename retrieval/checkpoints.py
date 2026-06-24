"""
Checkpoint utilities for the Paper2Fig-2026 Retrieval Framework.

Author: Nguyen Khanh
Project: Paper2Fig-2026
"""

import json
from pathlib import Path
from typing import Dict, Any


class CheckpointManager:
    """
    Save and load processing checkpoints.
    """

    def __init__(self, checkpoint_file: Path):

        self.checkpoint_file = checkpoint_file

    # --------------------------------------------------------

    def exists(self) -> bool:
        """
        Check whether checkpoint exists.
        """

        return self.checkpoint_file.exists()

    # --------------------------------------------------------

    def load(self) -> Dict[str, Any]:
        """
        Load checkpoint.

        Returns
        -------
        dict
        """

        if not self.exists():
            return {}

        with open(
            self.checkpoint_file,
            "r",
            encoding="utf-8",
        ) as f:

            return json.load(f)

    # --------------------------------------------------------

    def save(self, **kwargs) -> None:
        """
        Save checkpoint.

        Example
        -------
        save(
            processed=5000,
            batch=78
        )
        """

        with open(
            self.checkpoint_file,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                kwargs,
                f,
                indent=4,
            )

    # --------------------------------------------------------

    def reset(self) -> None:
        """
        Delete checkpoint.
        """

        if self.exists():

            self.checkpoint_file.unlink()