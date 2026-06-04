"""SGFADet network package.

This package exposes the paper-specific modules used by SGFADet while the
training engine remains compatible with the underlying Ultralytics semantic
segmentation pipeline.
"""

from .SGFADet_Model import SGFADetModel, build_sgfadet

__all__ = ["SGFADetModel", "build_sgfadet"]
