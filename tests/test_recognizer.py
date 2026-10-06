from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, ImageDraw

from formulaocr_offline.recognizer import (
    FormulaRecognitionError,
    OfflineFormulaOCR,
    extract_formula,
    looks_like_formula,
    normalize_formula,
    prepare_formula_image,
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


def test_formula_classifier_tolerates_capture_border():
    formula = Image.new("RGB", (290, 106), (246, 247, 245))
    draw = ImageDraw.Draw(formula)
    draw.rectangle((0, 0, 289, 105), outline=(93, 93, 92))
    draw.rectangle((40, 40, 250, 65), fill="black")
    assert looks_like_formula(formula)


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "L", "P", "CMYK", "LA"])
def test_prepare_formula_image_normalizes_dark_background(mode):
    formula = Image.new("RGB", (290, 106), (36, 36, 38))
    draw = ImageDraw.Draw(formula)
    draw.rectangle((0, 0, 289, 105), outline=(180, 180, 180))
    draw.rectangle((40, 40, 250, 65), fill=(230, 230, 230))
    formula = (
        formula.convert("P", palette=Image.Palette.ADAPTIVE)
        if mode == "P"
        else formula.convert(mode)
    )
    original = formula.tobytes()

    prepared = prepare_formula_image(formula)

    assert prepared.mode == "RGB"
    assert prepared.size == formula.size
    assert prepared.getpixel((20, 20)) == (255, 255, 255)
    assert prepared.getpixel((50, 50)) == (0, 0, 0)
    assert formula.tobytes() == original
    assert looks_like_formula(prepared)


def test_prepare_formula_image_normalizes_light_image_without_changing_source(tmp_path):
    formula = Image.new("RGB", (300, 100), (246, 247, 245))
    ImageDraw.Draw(formula).rectangle((40, 35, 260, 65), fill="black")
    image_path = tmp_path / "formula.png"
    formula.save(image_path)
    original = image_path.read_bytes()

    prepared = prepare_formula_image(image_path)
    assert prepared.getpixel((0, 0)) == (255, 255, 255)
    assert prepared.getpixel((50, 50)) == (0, 0, 0)
    assert image_path.read_bytes() == original


@pytest.mark.parametrize("foreground", [(0, 0, 0), (255, 255, 255), (230, 140, 40), (20, 60, 180)])
@pytest.mark.parametrize("alpha", [64, 128, 255])
def test_prepare_formula_image_handles_transparent_foreground(foreground, alpha):
    formula = Image.new("RGBA", (300, 100), (100, 50, 0, 0))
    ImageDraw.Draw(formula).rectangle((40, 35, 260, 65), fill=(*foreground, alpha))

    prepared = prepare_formula_image(formula)

    assert prepared.getpixel((0, 0)) == (255, 255, 255)
    assert prepared.getpixel((50, 50)) == (0, 0, 0)
    assert looks_like_formula(prepared)
    assert looks_like_formula(formula)


def test_prepare_formula_image_preserves_antialiasing():
    formula = Image.new("RGBA", (300, 100), (0, 0, 0, 0))
    draw = ImageDraw.Draw(formula)
    draw.rectangle((40, 35, 260, 65), fill="white")
    draw.line((40, 34, 260, 34), fill=(255, 255, 255, 128))

    assert prepare_formula_image(formula).getpixel((50, 34)) == (127, 127, 127)


@pytest.mark.parametrize(
    ("background", "foreground"),
    [
        ((250, 240, 210), (20, 60, 180)),
        ((20, 40, 70), (230, 140, 40)),
        ((200, 100, 100), (100, 200, 100)),
        ((120, 120, 120), (140, 140, 140)),
        ((150, 150, 150), (130, 130, 130)),
    ],
)
def test_prepare_formula_image_handles_color_and_low_contrast(background, foreground):
    formula = Image.new("RGB", (300, 100), background)
    ImageDraw.Draw(formula).rectangle((40, 35, 260, 65), fill=foreground)

    prepared = prepare_formula_image(formula)

    assert prepared.getpixel((0, 0)) == (255, 255, 255)
    assert prepared.getpixel((50, 50)) == (0, 0, 0)
    assert looks_like_formula(prepared)
    assert looks_like_formula(formula)


@pytest.mark.parametrize("with_formula", [True, False])
def test_prepare_formula_image_handles_gradient_background(with_formula):
    coordinates = np.indices((100, 300), dtype=np.float32)
    background = 35 + coordinates[0] * 0.5 + coordinates[1] * 0.3
    pixels = np.stack((background, background + 20, background + 40), axis=2).astype(np.uint8)
    formula = Image.fromarray(pixels)
    if with_formula:
        ImageDraw.Draw(formula).rectangle((40, 35, 260, 65), fill=(240, 220, 180))

    prepared = prepare_formula_image(formula)

    assert prepared.getpixel((20, 20))[0] >= 250
    assert prepared.getpixel((280, 80))[0] >= 250
    assert looks_like_formula(prepared) == with_formula
    if with_formula:
        assert prepared.getpixel((50, 50))[0] < 80


def test_prepare_formula_image_handles_jpeg_compression(tmp_path):
    formula = Image.new("RGB", (300, 100), (20, 40, 70))
    ImageDraw.Draw(formula).rectangle((40, 35, 260, 65), fill=(240, 200, 120))
    image_path = tmp_path / "formula.jpg"
    formula.save(image_path, quality=75)

    prepared = prepare_formula_image(image_path)

    assert prepared.getpixel((0, 0)) == (255, 255, 255)
    assert looks_like_formula(prepared)


def test_sparse_formula_strokes_survive_preprocessing():
    formula = Image.new("RGB", (300, 100), (20, 40, 70))
    ImageDraw.Draw(formula).line((100, 50, 140, 50), fill=(240, 200, 120))

    prepared = prepare_formula_image(formula)

    assert prepared.getpixel((120, 50)) == (0, 0, 0)
    assert looks_like_formula(formula)


def test_formula_classifier_rejects_normalized_texture():
    pixels = np.random.default_rng(42).integers(0, 256, (100, 300, 3), dtype=np.uint8)

    assert not looks_like_formula(prepare_formula_image(Image.fromarray(pixels)))


@pytest.mark.parametrize("classify", [True, False])
def test_recognize_normalizes_dark_image_before_inference(tmp_path, classify):
    formula = Image.new("RGB", (300, 100), (36, 38, 40))
    ImageDraw.Draw(formula).rectangle((40, 35, 260, 65), fill="white")
    image_path = tmp_path / "formula.png"
    formula.save(image_path)
    recognizer = OfflineFormulaOCR.__new__(OfflineFormulaOCR)
    inputs = []

    def predict(*, input, batch_size):
        inputs.append(input)
        assert batch_size == 1
        return [SimpleNamespace(json={"res": {"rec_formula": "x^2"}})]

    recognizer._model = SimpleNamespace(predict=predict)

    assert recognizer.recognize(image_path, classify=classify) == "x^2"
    assert len(inputs) == 1
    assert inputs[0].flags.c_contiguous
    np.testing.assert_array_equal(inputs[0][0, 0], [255, 255, 255])
    np.testing.assert_array_equal(inputs[0][50, 50], [0, 0, 0])


def test_recognize_rejects_blank_image_before_inference(tmp_path):
    image_path = tmp_path / "blank.png"
    Image.new("RGBA", (300, 100), (255, 255, 255, 0)).save(image_path)
    recognizer = OfflineFormulaOCR.__new__(OfflineFormulaOCR)

    with pytest.raises(FormulaRecognitionError, match="does not look like a formula"):
        recognizer.recognize(image_path)


def test_recognize_reports_invalid_image(tmp_path):
    image_path = tmp_path / "invalid.png"
    image_path.write_bytes(b"invalid image")
    recognizer = OfflineFormulaOCR.__new__(OfflineFormulaOCR)

    with pytest.raises(FormulaRecognitionError, match="Unable to read formula image"):
        recognizer.recognize(image_path)


def test_extract_formula():
    result = SimpleNamespace(json={"res": {"rec_formula": r"$x^2+y^2$"}})
    assert extract_formula(result) == "x^2+y^2"
