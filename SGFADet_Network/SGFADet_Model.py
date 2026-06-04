"""SGFADet model builder.

This wrapper makes the source tree explicit: SGFADet is the user-facing network,
while the lower-level training/inference runtime is provided by the modified
Ultralytics semantic segmentation engine bundled in this repository.
"""

from pathlib import Path
from typing import Any

from ultralytics import YOLO
from ultralytics.nn.tasks import SemanticSegmentationModel

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "SGFADet_Configs" / "Network" / "SGFADetn_Semantic.yaml"


class SGFADetModel(YOLO):
    """User-facing SGFADet model wrapper."""

    def __init__(self, model: str | Path = DEFAULT_CONFIG, **kwargs: Any) -> None:
        super().__init__(str(model), task="semantic", **kwargs)


def build_sgfadet(config: str | Path = DEFAULT_CONFIG, channels: int = 3, classes: int = 1, verbose: bool = False):
    """Build the raw SGFADet PyTorch module from its YAML configuration."""

    return SemanticSegmentationModel(str(config), ch=channels, nc=classes, verbose=verbose)
