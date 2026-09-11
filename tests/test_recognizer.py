from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw

from formulaocr_offline.recognizer import (
    FormulaRecognitionError,
    extract_formula,
    looks_like_formula,
    normalize_formula,
    require_model_dir,
    validate_formula,
)


def test_normalize_formula_wrappers():
    assert normalize_formula(r"$$\frac{1}{2}$$") == r"\frac{1}{2}"
    assert normalize_formula("```latex\nx+y\n```") == "x+y"


def test_validate_formula_rejects_unsafe_or_unbalanced_input():
    assert validate_formula(r"\frac{x}{y}") == r"\frac{x}{y}"
    with pytest.raises(FormulaRecognitionError, match="unbalanced"):
        validate_formula(r"\frac{x}{y")
    with pytest.raises(FormulaRecognitionError, match="disallowed"):
        validate_formula(r"\input{/etc/passwd}")


def test_require_model_dir(tmp_path):
    with pytest.raises(FormulaRecognitionError, match="model is missing"):
        require_model_dir(tmp_path)
    for name in ("inference.json", "inference.pdiparams", "inference.yml"):
        (tmp_path / name).touch()
    assert require_model_dir(tmp_path) == tmp_path.resolve()


def test_formula_classifier():
    blank = Image.new("RGB", (300, 100), "white")
    assert not looks_like_formula(blank)

    formula = blank.copy()
    draw = ImageDraw.Draw(formula)
    draw.rectangle((40, 35, 260, 65), fill="black")
    assert looks_like_formula(formula)


def test_extract_formula():
    result = SimpleNamespace(json={"res": {"rec_formula": r"$x^2+y^2$"}})
    assert extract_formula(result) == "x^2+y^2"
