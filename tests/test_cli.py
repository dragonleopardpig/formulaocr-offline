import json

from formulaocr_offline.cli import _format_output, build_parser


def test_minder_output():
    assert _format_output(r"\frac{1}{2}", True) == r"$$\frac{1}{2}$$"
    assert _format_output("x", False) == "x"


def test_parser_defaults():
    args = build_parser().parse_args(["formula.png"])
    assert str(args.image) == "formula.png"
    assert args.device == "auto"
    assert not args.no_classify


def test_worker_protocol_is_json_serializable():
    assert json.loads(json.dumps({"id": 1, "ok": True, "formula": "x"})) == {
        "id": 1,
        "ok": True,
        "formula": "x",
    }
