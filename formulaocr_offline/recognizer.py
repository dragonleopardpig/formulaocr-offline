"""High-accuracy, fully local mathematical formula recognition."""

from __future__ import annotations

import logging
import os
import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

MODEL_NAME = "PP-FormulaNet_plus-L"
MODEL_FILENAMES = ("inference.json", "inference.pdiparams", "inference.yml")
MAX_FORMULA_LENGTH = 32_768

_FORBIDDEN_COMMANDS = re.compile(
    r"\\(?:catcode|csname|documentclass|include|input|openin|openout|read|usepackage|write)\b",
    re.IGNORECASE,
)
_ENVIRONMENT = re.compile(r"\\(begin|end)\s*\{([^{}]+)\}")


class FormulaRecognitionError(RuntimeError):
    """Raised when an image cannot safely be converted to a formula."""


def default_model_dir() -> Path:
    """Return the configured local model directory."""
    configured = os.environ.get("FORMULA_OCR_MODEL_DIR")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".paddlex" / "official_models" / MODEL_NAME


def require_model_dir(model_dir: str | os.PathLike[str] | None = None) -> Path:
    """Resolve and verify a complete local model without downloading it."""
    path = Path(model_dir).expanduser() if model_dir is not None else default_model_dir()
    missing = [name for name in MODEL_FILENAMES if not (path / name).is_file()]
    if missing:
        raise FormulaRecognitionError(
            f"Offline model is missing from {path}. "
            "Use the Nix package or run `formulaocr-offline --download-model` once."
        )
    return path.resolve()


def _strip_math_wrapper(source: str) -> str:
    wrappers = (("$$", "$$"), (r"\[", r"\]"), (r"\(", r"\)"), ("$", "$"))
    for opening, closing in wrappers:
        if source.startswith(opening) and source.endswith(closing):
            return source[len(opening) : -len(closing)].strip()
    return source


def normalize_formula(source: str) -> str:
    """Remove presentation wrappers while preserving the expression."""
    formula = source.strip()
    if formula.startswith("```") and formula.endswith("```"):
        lines = formula.splitlines()
        if len(lines) >= 3:
            formula = "\n".join(lines[1:-1]).strip()
    return _strip_math_wrapper(formula).strip()


def validate_formula(source: str) -> str:
    """Reject malformed or unsafe TeX before returning model output."""
    formula = normalize_formula(source)
    if not formula:
        raise FormulaRecognitionError("The recognizer returned an empty formula")
    if len(formula) > MAX_FORMULA_LENGTH:
        raise FormulaRecognitionError("The recognized formula is unreasonably long")
    if any(ord(character) < 32 and character not in "\n\r\t" for character in formula):
        raise FormulaRecognitionError("The recognized formula contains control characters")
    if _FORBIDDEN_COMMANDS.search(formula):
        raise FormulaRecognitionError("The recognized formula contains a disallowed TeX command")

    brace_depth = 0
    escaped = False
    for character in formula:
        if escaped:
            escaped = False
            continue
        if character == "\\":
            escaped = True
        elif character == "{":
            brace_depth += 1
        elif character == "}":
            brace_depth -= 1
            if brace_depth < 0:
                break
    if brace_depth != 0:
        raise FormulaRecognitionError("The recognized formula has unbalanced braces")

    environments: list[str] = []
    for kind, name in _ENVIRONMENT.findall(formula):
        if kind == "begin":
            environments.append(name)
        elif not environments or environments.pop() != name:
            raise FormulaRecognitionError("The recognized formula has mismatched environments")
    if environments:
        raise FormulaRecognitionError("The recognized formula has an unclosed environment")
    return formula


def looks_like_formula(image: str | os.PathLike[str] | Image.Image) -> bool:
    """Conservatively distinguish a formula crop from a normal image."""
    if isinstance(image, Image.Image):
        picture = image.convert("RGB")
    else:
        with Image.open(image) as opened:
            picture = opened.convert("RGB")
    picture.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
    pixels = np.asarray(picture, dtype=np.float32)

    if pixels.shape[0] < 8 or pixels.shape[1] < 8:
        return False

    edge_width = max(1, min(pixels.shape[:2]) // 20)
    border = np.concatenate(
        (
            pixels[:edge_width].reshape(-1, 3),
            pixels[-edge_width:].reshape(-1, 3),
            pixels[:, :edge_width].reshape(-1, 3),
            pixels[:, -edge_width:].reshape(-1, 3),
        ),
        axis=0,
    )
    background = np.median(border, axis=0)
    gray = pixels.mean(axis=2)
    contrast = np.abs(gray - float(background.mean()))
    ink_density = float(np.mean(contrast > 35.0))
    background_share = float(np.mean(contrast < 20.0))
    color_spread = np.ptp(pixels, axis=2)

    return (
        0.001 <= ink_density <= 0.50
        and background_share >= 0.45
        and float(np.percentile(contrast, 99)) >= 60.0
        and float(np.percentile(color_spread, 95)) <= 45.0
    )


def extract_formula(result: Any) -> str:
    """Extract and validate LaTeX from a PaddleOCR result object."""
    payload = result.json
    if callable(payload):
        payload = payload()
    try:
        source = payload["res"]["rec_formula"]
    except (KeyError, TypeError) as error:
        raise FormulaRecognitionError("The recognizer returned an unexpected result") from error
    if not isinstance(source, str):
        raise FormulaRecognitionError("The recognizer did not return LaTeX text")
    return validate_formula(source)


def _quiet_inference_dependencies() -> None:
    """Keep stdout machine-readable and hide irrelevant inference warnings."""
    warnings.filterwarnings(
        "ignore",
        message="No ccache found.*",
        category=UserWarning,
    )
    logging.getLogger("paddlex").setLevel(logging.ERROR)


class OfflineFormulaOCR:
    """PP-FormulaNet wrapper that cannot download during recognition."""

    def __init__(
        self,
        model_dir: str | os.PathLike[str] | None = None,
        device: str = "auto",
    ) -> None:
        local_model = require_model_dir(model_dir)
        os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
        _quiet_inference_dependencies()

        try:
            import paddle
            from paddleocr import FormulaRecognition
        except ImportError as error:
            raise FormulaRecognitionError(
                "Recognition dependencies are not installed; use the Nix package "
                "or install the `runtime` project extra"
            ) from error

        logging.getLogger("paddlex").setLevel(logging.ERROR)
        if device == "auto":
            device = (
                "gpu"
                if paddle.device.is_compiled_with_cuda()
                and paddle.device.cuda.device_count() > 0
                else "cpu"
            )
        if device not in {"cpu", "gpu"}:
            raise FormulaRecognitionError(f"Unsupported inference device: {device}")

        try:
            self._model = FormulaRecognition(
                model_name=MODEL_NAME,
                model_dir=str(local_model),
                device=device,
            )
        except Exception as error:
            raise FormulaRecognitionError(f"Unable to load the offline model: {error}") from error

    def recognize(
        self,
        image: str | os.PathLike[str],
        *,
        classify: bool = True,
    ) -> str:
        """Recognize one local image and return validated LaTeX."""
        image_path = Path(image).expanduser()
        if not image_path.is_file():
            raise FormulaRecognitionError(f"Image does not exist: {image_path}")
        if classify and not looks_like_formula(image_path):
            raise FormulaRecognitionError("The image does not look like a formula crop")
        try:
            results = self._model.predict(input=str(image_path.resolve()), batch_size=1)
        except Exception as error:
            raise FormulaRecognitionError(f"Formula recognition failed: {error}") from error
        if len(results) != 1:
            raise FormulaRecognitionError("The recognizer did not return exactly one result")
        return extract_formula(results[0])


def download_model(device: str = "cpu") -> Path:
    """Explicitly download the official model for later offline use."""
    try:
        return require_model_dir()
    except FormulaRecognitionError:
        pass

    os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", "bos")
    try:
        from paddleocr import FormulaRecognition
    except ImportError as error:
        raise FormulaRecognitionError(
            "Recognition dependencies are not installed; install the `runtime` project extra"
        ) from error
    FormulaRecognition(model_name=MODEL_NAME, device=device)
    return require_model_dir()
