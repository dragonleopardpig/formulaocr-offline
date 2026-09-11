# formulaocr-offline

`formulaocr-offline` converts a cropped image of a mathematical expression to
LaTeX entirely on the local machine. It uses PaddleOCR's
PP-FormulaNet-plus-L model and validates the generated TeX before printing it.

The project was extracted from the offline recognition work in
[`dragonleopardpig/LaTeX-OCR`](https://github.com/dragonleopardpig/LaTeX-OCR)
so it can be installed without the pix2tex GUI, API, training stack, or a
mutable development environment.

## Run with Nix

The first build downloads the pinned 698 MB model archive. Recognition makes
no network requests afterward.

```console
nix run github:dragonleopardpig/formulaocr-offline -- formula.png
```

Useful options:

```console
formulaocr-offline --minder formula.png       # output $$...$$
formulaocr-offline --copy formula.png         # copy to Wayland clipboard
formulaocr-offline --device cpu formula.png
formulaocr-offline --worker                    # newline-delimited JSON mode
```

## NixOS module

```nix
{
  inputs.formulaocr-offline = {
    url = "github:dragonleopardpig/formulaocr-offline";
    inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = { formulaocr-offline, ... }: {
    nixosConfigurations.my-host = nixpkgs.lib.nixosSystem {
      modules = [ formulaocr-offline.nixosModules.default ];
    };
  };
}
```

The module installs the command and sets `MINDER_FORMULA_OCR` to its immutable
Nix store path. The default overlay and package are also exported for custom
configurations.

## Non-Nix installation

```console
python -m pip install '.[runtime]'
formulaocr-offline --download-model
formulaocr-offline formula.png
```

Model downloading is an explicit setup operation. Normal recognition refuses
to proceed unless all model files already exist locally. Set
`FORMULA_OCR_MODEL_DIR` to use a different model location.

## Development

```console
nix develop
python -m pytest
ruff check .
nix flake check
```

Project code is MIT-licensed. PaddleOCR, PaddleX, PaddlePaddle, and their model
artifacts retain their respective upstream licenses.
