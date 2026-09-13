"""Offline mathematical formula recognition."""

from .recognizer import (
    FormulaRecognitionError,
    OfflineFormulaOCR,
    download_model,
    looks_like_formula,
    normalize_formula,
    validate_formula,
)

__all__ = [
    "FormulaRecognitionError",
    "OfflineFormulaOCR",
    "download_model",
    "looks_like_formula",
    "normalize_formula",
    "validate_formula",
]
__version__ = "0.1.2"
