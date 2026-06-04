"""Multi-scale feature aggregation neck description for SGFADet.

The executable neck is specified in ``SGFADet_Configs/Network/SGFADetn_Semantic.yaml``:
P5 is upsampled and concatenated with P4, then the result is upsampled and
concatenated with P3. C3k2 blocks refine the aggregated features before the
ATAH semantic head receives P3/P4/P5 feature maps.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SGFADetNeckDescription:
    """Static description of the SGFADet neck used for documentation and imports."""

    stages: tuple[str, ...] = (
        "Upsample P5 and concatenate with SAF-fused P4",
        "Refine P4 aggregation with C3k2",
        "Upsample refined P4 and concatenate with SAF-fused P3",
        "Refine P3 aggregation with C3k2",
        "Forward P3/P4/P5 to ATAHSegment",
    )
