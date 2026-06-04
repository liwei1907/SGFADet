"""RGB backbone calibration components for SGFADet.

The RGB backbone itself is defined in the SGFADet YAML with Conv, C3k2, SPPF
and C2PSA blocks. This file exposes SFC, the salient feature calibrator embedded
into the RGB branch at key P3/P4/P5 levels.
"""

from ultralytics.nn.modules.sgfadet import SFC

__all__ = ["SFC"]
