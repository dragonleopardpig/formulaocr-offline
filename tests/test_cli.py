import json

from formulaocr_offline import cli
from formulaocr_offline.cli import _format_output, build_parser


def test_minder_output():
    assert _format_output(r"\frac{1}{2}", True) == r"$$\frac{1}{2}$$"
    assert _format_output("x", False) == "x"


def test_parser_defaults():
    args = build_parser().parse_args(["formula.png"])
    assert str(args.image) == "formula.png"
    assert args.device == "auto"
    assert args.cpu_threads is None
    assert not args.no_classify


def test_cpu_threads_option_reaches_the_recognizer(monkeypatch, capsys):
    loaded = []

    class Recognizer:
        def __init__(self, **options):
            loaded.append(options)

        def recognize(self, image, *, classify):
            return "x^2"

    monkeypatch.setattr(cli, "OfflineFormulaOCR", Recognizer)

    assert cli.main(["--cpu-threads", "2", "formula.png"]) == 0
    assert capsys.readouterr().out == "x^2\n"
    assert loaded[0]["cpu_threads"] == 2


def test_worker_protocol_is_json_serializable():
    assert json.loads(json.dumps({"id": 1, "ok": True, "formula": "x"})) == {
        "id": 1,
        "ok": True,
        "formula": "x",
    }
