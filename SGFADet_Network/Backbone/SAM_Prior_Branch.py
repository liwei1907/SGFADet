"""Frozen SAM prior branch for SGFADet.

The branch extracts P3/P4/P5 semantic priors from a frozen SAM image encoder
when a SAM checkpoint is provided. For code verification without SAM weights it
falls back to a frozen RGB+Sobel prior, preserving the same tensor interface.
"""

from ultralytics.nn.modules.sgfadet import SAMFrozenPyramid

__all__ = ["SAMFrozenPyramid"]
