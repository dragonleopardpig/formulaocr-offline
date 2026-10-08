"""High-accuracy, fully local mathematical formula recognition."""

from __future__ import annotations

import logging
import os
import re
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

MODEL_NAME = "PP-FormulaNet_plus-L"
MODEL_FILENAMES = ("inference.json", "inference.pdiparams", "inference.yml")
MAX_FORMULA_LENGTH = 32_768

# PP-FormulaNet runs no faster on more CPU threads than this, and PaddleX's own
# default of 10 makes recognition several times slower when the machine is busy.
MAX_CPU_THREADS = 4

# Background estimation: pixels sampled, trial planes drawn, and the color
# difference (in 8-bit levels) below which a pixel still counts as background.
_BACKGROUND_SAMPLES = 4096
_BACKGROUND_TRIALS = 256
_BACKGROUND_TOLERANCE = 2.0

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


def _sheet_color(pixels: np.ndarray) -> tuple[int, int, int]:
    """Choose the color laid behind the transparent pixels of an RGBA image."""
    alpha = pixels[:, :, 3]
    opaque = alpha >= 128
    if 2 * np.count_nonzero(opaque) >= opaque.size:
        # A mostly opaque picture: stray transparent pixels take its own background.
        color = np.median(pixels[opaque, :3], axis=0)
        return tuple(int(value) for value in color)
    # Ink on a transparent sheet: lay a contrasting sheet behind it.
    ink = pixels[alpha > 0, :3]
    light_ink = bool(ink.size) and float(np.median(ink, axis=0).mean()) > 128
    return (0, 0, 0) if light_ink else (255, 255, 255)


def _fit_background(pixels: np.ndarray) -> np.ndarray:
    """Return the flat color or linear gradient that most of an RGB image lies on.

    The image edges are not assumed to be background: in a tight crop the ink
    touches them. The plane is the least-median-of-residuals fit over the whole
    picture, which stays on the background while ink covers less than half of it.
    """
    height, width = pixels.shape[:2]
    step = max(1, int(np.ceil(np.sqrt(height * width / _BACKGROUND_SAMPLES))))
    rows, columns = np.meshgrid(
        np.arange(min(step // 2, height - 1), height, step),
        np.arange(min(step // 2, width - 1), width, step),
        indexing="ij",
    )
    samples = pixels[rows, columns].reshape(-1, 3)
    design = np.stack(
        (np.ones(rows.size), rows.ravel() / height, columns.ravel() / width),
        axis=1,
    ).astype(np.float32)

    def residuals(planes: np.ndarray) -> np.ndarray:
        return np.abs(design @ planes - samples).max(axis=-1)

    flat = np.zeros((1, 3, 3), dtype=np.float32)
    flat[0, 0] = np.median(samples, axis=0)
    picks = np.random.default_rng(0).integers(0, len(samples), (_BACKGROUND_TRIALS, 3))
    tilted = np.linalg.pinv(design[picks]) @ samples[picks]
    candidates = np.concatenate((flat, tilted))
    candidate_residuals = residuals(candidates)
    best = int(np.argmin(np.median(candidate_residuals, axis=1)))
    plane, residual = candidates[best], candidate_residuals[best]

    for _ in range(2):
        # Keep what lies within 2.5 standard deviations, judged by the median residual.
        spread = 2.5 * 1.4826 * float(np.median(residual))
        inliers = residual <= max(_BACKGROUND_TOLERANCE, spread)
        plane = np.linalg.lstsq(design[inliers], samples[inliers], rcond=None)[0]
        plane[0] += np.median((samples - design @ plane)[inliers], axis=0)
        residual = residuals(plane)

    row_share = np.arange(height, dtype=np.float32)[:, None, None] / height
    column_share = np.arange(width, dtype=np.float32)[None, :, None] / width
    return np.clip(plane[0] + row_share * plane[1] + column_share * plane[2], 0, 255)


def prepare_formula_image(image: str | os.PathLike[str] | Image.Image) -> Image.Image:
    """Normalize formula contrast across colored, shaded and transparent backgrounds."""
    if isinstance(image, Image.Image):
        picture = ImageOps.exif_transpose(image).convert("RGBA")
    else:
        with Image.open(image) as opened:
            picture = ImageOps.exif_transpose(opened).convert("RGBA")

    sheet = Image.new("RGBA", picture.size, _sheet_color(np.asarray(picture)))
    colors = Image.alpha_composite(sheet, picture).convert("RGB")
    pixels = np.asarray(colors, dtype=np.float32)
    contrast = np.max(np.abs(pixels - _fit_background(pixels)), axis=2)
    foreground_contrast = contrast[contrast > _BACKGROUND_TOLERANCE]
    if not foreground_contrast.size:
        return Image.new("RGB", picture.size, "white")
    scale = float(np.percentile(foreground_contrast, 99))
    normalized = np.rint(255 * (1 - np.clip(contrast / scale, 0, 1))).astype(np.uint8)
    return Image.fromarray(normalized).convert("RGB")


def looks_like_formula(image: str | os.PathLike[str] | Image.Image) -> bool:
    """Conservatively distinguish a formula crop from a normal image."""
    return _has_formula_contrast(prepare_formula_image(image))


def _has_formula_contrast(picture: Image.Image) -> bool:
    """Judge a prepared image: ink against the background that fills most of it."""
    picture = picture.copy()
    picture.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
    pixels = np.asarray(picture, dtype=np.float32)

    if pixels.shape[0] < 8 or pixels.shape[1] < 8:
        return False

    gray = pixels.mean(axis=2)
    contrast = np.abs(gray - float(np.median(gray)))
    ink_density = float(np.mean(contrast > 35.0))
    background_share = float(np.mean(contrast < 20.0))

    return 0.001 <= ink_density <= 0.50 and background_share >= 0.45


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


@contextmanager
def _fast_config_parsing() -> Iterator[None]:
    """Let PaddleX parse the model configuration with libyaml when it can.

    PaddleX reads the 100,000-line ``inference.yml`` through PyYAML's pure-Python
    ``FullLoader``, which costs several seconds on every start. ``CFullLoader``
    builds the same objects with the C parser.
    """
    try:
        import yaml

        fast_loader = yaml.CFullLoader
    except (ImportError, AttributeError):
        yield
        return
    python_loader, yaml.FullLoader = yaml.FullLoader, fast_loader
    try:
        yield
    finally:
        yaml.FullLoader = python_loader


def _default_cpu_threads() -> int:
    """Return the CPU inference thread count used when none is requested."""
    try:
        usable = len(os.sched_getaffinity(0))
    except AttributeError:
        usable = os.cpu_count() or 1
    return max(1, min(MAX_CPU_THREADS, usable))


class OfflineFormulaOCR:
    """PP-FormulaNet wrapper that cannot download during recognition."""

    def __init__(
        self,
        model_dir: str | os.PathLike[str] | None = None,
        device: str = "auto",
        cpu_threads: int | None = None,
    ) -> None:
        local_model = require_model_dir(model_dir)
        if cpu_threads is None:
            cpu_threads = _default_cpu_threads()
        if cpu_threads < 1:
            raise FormulaRecognitionError("The CPU thread count must be at least 1")
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
                if paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0
                else "cpu"
            )
        if device not in {"cpu", "gpu"}:
            raise FormulaRecognitionError(f"Unsupported inference device: {device}")

        try:
            with _fast_config_parsing():
                self._model = FormulaRecognition(
                    model_name=MODEL_NAME,
                    model_dir=str(local_model),
                    device=device,
                    cpu_threads=cpu_threads,
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
        try:
            picture = prepare_formula_image(image_path)
        except (OSError, ValueError) as error:
            raise FormulaRecognitionError(f"Unable to read formula image: {error}") from error
        if classify and not _has_formula_contrast(picture):
            raise FormulaRecognitionError("The image does not look like a formula crop")
        try:
            pixels = np.asarray(picture)[:, :, ::-1].copy()
            results = self._model.predict(input=pixels, batch_size=1)
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
