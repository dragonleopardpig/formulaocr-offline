import os
import sys
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
    tidy_token_spacing,
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


def _draw_tight_crop(draw, ink):
    """Draw a formula cropped with no margin, so ink touches every image edge."""
    draw.rectangle((0, 20, 5, 80), fill=ink)  # delimiter against the left edge
    draw.rectangle((294, 10, 299, 40), fill=ink)  # superscript against the right edge
    draw.rectangle((0, 48, 299, 51), fill=ink)  # fraction bar across the full width
    draw.rectangle((140, 0, 160, 30), fill=ink)  # glyph against the top edge
    draw.rectangle((100, 70, 120, 99), fill=ink)  # glyph against the bottom edge


def _tight_crop_strokes():
    strokes = Image.new("RGB", (300, 100), "white")
    _draw_tight_crop(ImageDraw.Draw(strokes), "black")
    return np.asarray(strokes)


@pytest.mark.parametrize(
    ("background", "ink"),
    [
        ((255, 255, 255), (0, 0, 0)),
        ((30, 30, 30), (212, 212, 212)),
        ((250, 240, 210), (20, 60, 180)),
    ],
)
def test_prepare_formula_image_handles_ink_touching_the_edges(background, ink):
    formula = Image.new("RGB", (300, 100), background)
    _draw_tight_crop(ImageDraw.Draw(formula), ink)

    np.testing.assert_array_equal(np.asarray(prepare_formula_image(formula)), _tight_crop_strokes())
    assert looks_like_formula(formula)


@pytest.mark.parametrize("ink", [(0, 0, 0), (255, 255, 255)])
def test_prepare_formula_image_handles_tight_transparent_crop(ink):
    formula = Image.new("RGBA", (300, 100), (0, 0, 0, 0))
    _draw_tight_crop(ImageDraw.Draw(formula), (*ink, 255))

    np.testing.assert_array_equal(np.asarray(prepare_formula_image(formula)), _tight_crop_strokes())
    assert looks_like_formula(formula)


def test_prepare_formula_image_handles_tight_crop_on_gradient():
    coordinates = np.indices((100, 300), dtype=np.float32)
    background = 35 + coordinates[0] * 0.5 + coordinates[1] * 0.3
    pixels = np.stack((background, background + 20, background + 40), axis=2).astype(np.uint8)
    formula = Image.fromarray(pixels)
    _draw_tight_crop(ImageDraw.Draw(formula), (240, 220, 180))

    prepared = np.asarray(prepare_formula_image(formula))[:, :, 0]

    assert prepared[_tight_crop_strokes()[:, :, 0] == 255].min() >= 250
    assert prepared[50, 2] < 80
    assert prepared[49, 150] < 128
    assert looks_like_formula(formula)


def test_prepare_formula_image_ignores_frame_around_small_capture():
    formula = Image.new("RGB", (120, 30), (36, 36, 38))
    draw = ImageDraw.Draw(formula)
    draw.rectangle((0, 0, 119, 29), outline=(180, 180, 180))
    draw.rectangle((20, 10, 100, 20), fill=(230, 230, 230))

    prepared = prepare_formula_image(formula)

    assert prepared.getpixel((10, 5)) == (255, 255, 255)
    assert prepared.getpixel((50, 15)) == (0, 0, 0)
    assert looks_like_formula(formula)


def test_prepare_formula_image_fills_transparent_corners_with_background():
    formula = Image.new("RGBA", (300, 100), (36, 36, 38, 255))
    draw = ImageDraw.Draw(formula)
    draw.rectangle((40, 35, 260, 65), fill=(230, 230, 230, 255))
    for left, top in ((0, 0), (292, 0), (0, 92), (292, 92)):
        draw.rectangle((left, top, left + 7, top + 7), fill=(0, 0, 0, 0))

    prepared = prepare_formula_image(formula)

    assert prepared.getpixel((0, 0)) == (255, 255, 255)
    assert prepared.getpixel((50, 50)) == (0, 0, 0)


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


@pytest.fixture
def loaded_models(tmp_path, monkeypatch):
    """Stand in for the inference runtime and record how each model is loaded."""
    loaded = []

    def load(**options):
        loaded.append(options)
        return SimpleNamespace()

    cuda = SimpleNamespace(device_count=lambda: 0)
    device = SimpleNamespace(is_compiled_with_cuda=lambda: False, cuda=cuda)
    monkeypatch.setitem(sys.modules, "paddle", SimpleNamespace(device=device))
    monkeypatch.setitem(sys.modules, "paddleocr", SimpleNamespace(FormulaRecognition=load))
    monkeypatch.setenv("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    for name in ("inference.json", "inference.pdiparams", "inference.yml"):
        (tmp_path / name).touch()
    return loaded


@pytest.mark.parametrize(("usable_cpus", "expected"), [(16, 4), (2, 2), (1, 1)])
def test_cpu_inference_threads_are_capped_by_default(
    tmp_path, monkeypatch, loaded_models, usable_cpus, expected
):
    monkeypatch.setattr(
        os, "sched_getaffinity", lambda _pid: set(range(usable_cpus)), raising=False
    )

    OfflineFormulaOCR(model_dir=tmp_path)

    assert loaded_models[0]["device"] == "cpu"
    assert loaded_models[0]["cpu_threads"] == expected


def test_cpu_inference_threads_can_be_requested(tmp_path, loaded_models):
    OfflineFormulaOCR(model_dir=tmp_path, cpu_threads=6)
    assert loaded_models[0]["cpu_threads"] == 6

    with pytest.raises(FormulaRecognitionError, match="at least 1"):
        OfflineFormulaOCR(model_dir=tmp_path, cpu_threads=0)
    assert len(loaded_models) == 1


@pytest.mark.parametrize("loads", [True, False])
def test_model_configuration_is_parsed_with_libyaml(tmp_path, monkeypatch, loaded_models, loads):
    yaml = pytest.importorskip("yaml")
    if not hasattr(yaml, "CFullLoader"):
        pytest.skip("PyYAML was built without libyaml")
    python_loader = yaml.FullLoader
    loaders = []

    def load(**options):
        # PaddleX reads inference.yml with whatever yaml.FullLoader is right now.
        loaders.append(yaml.FullLoader)
        if not loads:
            raise RuntimeError("broken model")
        return SimpleNamespace()

    monkeypatch.setitem(sys.modules, "paddleocr", SimpleNamespace(FormulaRecognition=load))

    if loads:
        OfflineFormulaOCR(model_dir=tmp_path)
    else:
        with pytest.raises(FormulaRecognitionError, match="broken model"):
            OfflineFormulaOCR(model_dir=tmp_path)

    assert loaders == [yaml.CFullLoader]
    assert yaml.FullLoader is python_loader


def test_token_spacing_keeps_a_command_apart_from_the_letter_after_it():
    decoded = (
        r"\mathbf { \nabla \times A } = \left( \frac { 1 } { r } "
        r"\frac { \partial A _ { z } } { \partial \theta } - "
        r"\frac { \partial A _ { \theta } } { \partial z } \right) \mathbf { r }"
    )

    assert tidy_token_spacing(decoded) == (
        r"\mathbf{\nabla\times A}=\left(\frac{1}{r}"
        r"\frac{\partial A_{z}}{\partial\theta}-"
        r"\frac{\partial A_{\theta}}{\partial z}\right)\mathbf{r}"
    )


@pytest.mark.parametrize(
    ("decoded", "tidy"),
    [
        (
            r"\mathbf { A } = A _ { x } \mathbf { i } + A _ { y } \mathbf { j }",
            r"\mathbf{A}=A_{x}\mathbf{i}+A_{y}\mathbf{j}",
        ),
        (
            r"\int _ { 0 } ^ { \infty } e ^ { - x ^ { 2 } }   d x = { \frac { \pi } { 2 } }",
            r"\int_{0}^{\infty}e^{-x^{2}}d x={\frac{\pi}{2}}",
        ),
        (r"a \ b \, c \\ d", r"a\ b\,c\\ d"),
    ],
)
def test_token_spacing_removes_the_spaces_between_tokens(decoded, tidy):
    assert tidy_token_spacing(decoded) == tidy


def _model_with_cleanup(monkeypatch, cleanup):
    decoder = SimpleNamespace(normalize=cleanup)
    model = SimpleNamespace(paddlex_predictor=SimpleNamespace(post_op=decoder))
    runtime = SimpleNamespace(FormulaRecognition=lambda **options: model)
    monkeypatch.setitem(sys.modules, "paddleocr", runtime)
    return decoder


def test_spacing_cleanup_that_glues_commands_to_letters_is_replaced(
    tmp_path, monkeypatch, loaded_models
):
    # PaddleX 3.4 strips every space once a \mathbf group holds a command and a letter.
    decoder = _model_with_cleanup(monkeypatch, lambda decoded: decoded.replace(" ", ""))

    OfflineFormulaOCR(model_dir=tmp_path)

    assert decoder.normalize(r"\frac { \partial A } { \partial z }") == (
        r"\frac{\partial A}{\partial z}"
    )


def test_sound_spacing_cleanup_is_left_alone(tmp_path, monkeypatch, loaded_models):
    def sound(decoded):
        return tidy_token_spacing(decoded)

    def failing(decoded):
        raise ValueError(decoded)

    for cleanup in (sound, failing):
        decoder = _model_with_cleanup(monkeypatch, cleanup)
        OfflineFormulaOCR(model_dir=tmp_path)
        assert decoder.normalize is cleanup
