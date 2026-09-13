# formulaocr-offline

`formulaocr-offline` converts a cropped image of a mathematical expression to
editable LaTeX entirely on the local machine. It uses PaddleOCR's
[`PP-FormulaNet_plus-L`](https://huggingface.co/PaddlePaddle/PP-FormulaNet_plus-L)
model and validates the generated TeX before printing it.

The project was extracted from the offline recognition work in
[`dragonleopardpig/LaTeX-OCR`](https://github.com/dragonleopardpig/LaTeX-OCR)
so that formula recognition can be installed without the pix2tex GUI, an API
service, training dependencies, or a mutable development environment.

## Features

- Runs inference locally after the one-time installation and model download.
- Accepts PNG, JPEG, and other image formats supported by Pillow.
- Returns plain LaTeX suitable for scripts and editor integrations.
- Can wrap output in Minder's `$$...$$` delimiters.
- Can copy output to a Wayland clipboard with `wl-copy`.
- Offers a persistent newline-delimited JSON worker for integrations.
- Rejects malformed output and potentially unsafe TeX file/process commands.
- Refuses to download a model implicitly during normal recognition.

For best results, provide a tightly cropped image containing one formula on a
plain, high-contrast background. OCR output should still be reviewed before it
is used in a publication or calculation.

## Requirements

- A 64-bit Linux system is the tested platform.
- Python 3.10 through 3.13 for a non-Nix installation. Python 3.11 or 3.12 is
  recommended for broad binary-wheel compatibility.
- Approximately 698 MB for the model, plus the Python inference dependencies.
- `wl-copy` from `wl-clipboard` only when using the optional `--copy` flag.

The Nix package currently supports `x86_64-linux`. The Python package may work
on other platforms supported by PaddlePaddle, but those configurations are not
currently tested here.

On NixOS, use the Nix package below or the provided devenv environment. A
regular `python -m venv` followed by `pip install` can install manylinux wheels
whose C/C++ libraries are not visible through NixOS library paths.

## Quick start with Nix

Run directly from the GitHub flake:

```console
nix run github:dragonleopardpig/formulaocr-offline -- formula.png
```

The first Nix build downloads and verifies the pinned official model archive.
The completed package contains an immutable model path, and recognition makes
no network requests. Do not create or activate a Python virtual environment for
this installation method, and do not separately download the model.

To install the command in a Nix profile:

```console
nix profile install github:dragonleopardpig/formulaocr-offline
formulaocr-offline formula.png
```

## Source checkout on NixOS with devenv

The repository includes a devenv configuration for developing or running the
Python installation on NixOS. It uses Python 3.12, enables manylinux library
compatibility for binary wheels, installs all runtime and development extras
with `uv`, and provides the pinned model through the Nix store.

```console
git clone https://github.com/dragonleopardpig/formulaocr-offline.git
cd formulaocr-offline
devenv shell
formulaocr-offline --version
formulaocr-offline --download-model
```

The final command verifies the configured model and prints its Nix store path;
it does not download a second copy into `~/.paddlex`. The first `devenv shell`
can take several minutes because it installs the Python inference dependencies.
The environment and virtual environment state live under `.devenv`; remove
that directory if the checkout no longer needs its local environment state.

For ordinary use rather than source development, prefer the Nix package from
the previous section. Install devenv by following its
[official installation instructions](https://devenv.sh/getting-started/).

## General installation on non-Nix Linux

The project is not currently published on PyPI and does not provide prebuilt
GitHub release binaries. Install it from this repository in an isolated Python
virtual environment. These instructions target conventional distributions such
as Debian, Ubuntu, Fedora, and Arch Linux; they are not the NixOS installation
instructions.

### 1. Install system prerequisites

Install Git, Python, pip, and Python virtual-environment support. For example,
on Debian or Ubuntu:

```console
sudo apt update
sudo apt install curl git libgl1 libglib2.0-0 libstdc++6 python3 python3-pip python3-venv
```

On Fedora:

```console
sudo dnf install curl git glib2 libglvnd-glx libstdc++ python3 python3-pip
```

On Arch Linux:

```console
sudo pacman -S curl gcc-libs git glib2 libglvnd python
```

### 2. Clone the project and create a virtual environment

```console
git clone https://github.com/dragonleopardpig/formulaocr-offline.git
cd formulaocr-offline
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

Activate the same environment with `source .venv/bin/activate` whenever a new
shell needs to use its `formulaocr-offline` command.

### 3. Install the application and CPU inference runtime

```console
python -m pip install '.[runtime]'
```

This installs the project together with PaddleOCR, the CPU PaddlePaddle runtime,
Pillow, OpenCV, and the other required Python libraries. The environment can be
removed cleanly by deleting the cloned directory. The separately downloaded
model cache under `~/.paddlex` must be removed separately if it is no longer
needed.

Alternatively, modern pip versions can install the tagged source directly into
an already-active virtual environment:

```console
python -m pip install \
  'formulaocr-offline[runtime] @ git+https://github.com/dragonleopardpig/formulaocr-offline.git@v0.1.2'
```

### 4. Download the model once

```console
formulaocr-offline --download-model
```

PaddleOCR downloads the official model to:

```text
~/.paddlex/official_models/PP-FormulaNet_plus-L
```

Normal recognition never downloads a missing model automatically. This makes a
missing or offline setup fail clearly instead of initiating unexpected network
traffic. To keep the model elsewhere, set `FORMULA_OCR_MODEL_DIR` or use
`--model-dir`:

```console
export FORMULA_OCR_MODEL_DIR=/path/to/PP-FormulaNet_plus-L
formulaocr-offline formula.png
```

The selected directory must contain `inference.json`, `inference.pdiparams`,
and `inference.yml` from a compatible `PP-FormulaNet_plus-L` inference model.

### 5. Verify the installation

```console
formulaocr-offline --version
curl -L -o formula.png \
  https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_formula_rec_001.png
formulaocr-offline formula.png
```

The last command should print recognized LaTeX to standard output.

## GPU installation

The `runtime` extra installs the CPU package `paddlepaddle`. For NVIDIA GPU
inference, remove that package and install the compatible `paddlepaddle-gpu`
wheel for the installed driver, following the
[official PaddlePaddle installation guide](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/paddlepaddle_installation.en.md):

```console
python -m pip uninstall paddlepaddle
# Install the appropriate paddlepaddle-gpu wheel from the official guide.
formulaocr-offline --download-model --device gpu
formulaocr-offline --device gpu formula.png
```

Do not install `paddlepaddle` and `paddlepaddle-gpu` together in the same
virtual environment. GPU support depends on the operating system, architecture,
NVIDIA driver, and the wheels currently published by PaddlePaddle.

## Command-line usage

```console
formulaocr-offline [OPTIONS] IMAGE
```

Common examples:

```console
# Print plain LaTeX.
formulaocr-offline formula.png

# Print LaTeX wrapped for Minder as $$...$$.
formulaocr-offline --minder formula.png

# Copy plain LaTeX to the Wayland clipboard and also print it.
formulaocr-offline --copy formula.png

# Force CPU or GPU inference instead of automatic selection.
formulaocr-offline --device cpu formula.png
formulaocr-offline --device gpu formula.png

# Use an explicitly installed model directory.
formulaocr-offline --model-dir /path/to/model formula.png

# Bypass the conservative formula-image classifier after a false rejection.
formulaocr-offline --no-classify formula.png
```

Run `formulaocr-offline --help` for the complete option list.

### Clipboard output

`--copy` requires `wl-copy`, available from the `wl-clipboard` package on most
Linux distributions:

```console
sudo apt install wl-clipboard       # Debian/Ubuntu
sudo dnf install wl-clipboard       # Fedora
sudo pacman -S wl-clipboard         # Arch Linux
```

This option targets Wayland. Without it, the recognized formula is always
written to standard output and can be piped to another clipboard tool.

### Worker mode

Worker mode loads the model once and accepts one JSON request per input line.
Image paths should be absolute:

```console
printf '%s\n' '{"id":1,"image":"/absolute/path/formula.png"}' | \
  formulaocr-offline --worker
```

It first emits `{"ready": true}` and then one response per request:

```json
{"id": 1, "ok": true, "formula": "x^2+y^2"}
```

Failures are returned without terminating the worker:

```json
{"id": 1, "ok": false, "error": "Image does not exist: ..."}
```

## Minder integration

Minder looks for `formulaocr-offline` on `PATH`. A specific executable can be
selected with `MINDER_FORMULA_OCR`:

```console
export MINDER_FORMULA_OCR="$HOME/path/to/formulaocr-offline/.venv/bin/formulaocr-offline"
minder
```

For a desktop launcher, set `MINDER_FORMULA_OCR` in the desktop session's
environment or use the NixOS module below. The exact image-to-LaTeX shortcut
depends on the installed Minder build.

### Using another OCR-to-LaTeX package with Minder

Minder is not restricted to this project or to `PP-FormulaNet_plus-L`. A user
may install another local OCR-to-LaTeX model or package, subject to that
project's software and model licenses, and select an adapter executable with
`MINDER_FORMULA_OCR`.

For example, PaddleOCR publishes several other formula-recognition models in
its [official model list](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/model_list.md),
and projects such as [LaTeX-OCR/pix2tex](https://github.com/lukas-blecher/LaTeX-OCR)
provide a different recognition stack. Minder does not require a particular
model internally; it only requires the command-line contract below. An online
OCR service could also be hidden behind an adapter, but then recognition would
no longer be private or offline.

The executable contract expected by Minder is:

1. Receive the local image path as its only command-line argument.
2. Write one unwrapped LaTeX expression to standard output. Do not add `$`,
   `$$`, `\\(`, or `\\[` delimiters.
3. Write diagnostics to standard error, not standard output.
4. Exit with status 0 on success and a nonzero status on failure.
5. Finish within 30 seconds and return no more than 32 KiB of output.

If another recognizer needs different arguments or produces JSON, place a
small adapter script in front of it and point Minder to that script:

```console
export MINDER_FORMULA_OCR=/absolute/path/to/my-formula-ocr-adapter
minder
```

The `--model-dir` option of `formulaocr-offline` does **not** make arbitrary
model architectures interchangeable. It expects files compatible with the
PaddleOCR `PP-FormulaNet_plus-L` implementation. A different model normally
requires its own runner or adapter.

## NixOS module

Add the repository as a flake input and import its module:

```nix
{
  inputs.formulaocr-offline = {
    url = "github:dragonleopardpig/formulaocr-offline";
    inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = { formulaocr-offline, nixpkgs, ... }: {
    nixosConfigurations.my-host = nixpkgs.lib.nixosSystem {
      system = "x86_64-linux";
      modules = [ formulaocr-offline.nixosModules.default ];
    };
  };
}
```

The module installs the command and sets `MINDER_FORMULA_OCR` to its immutable
Nix store path. The flake also exports `overlays.default` and
`packages.x86_64-linux.formulaocr-offline` for custom configurations.

## Troubleshooting

### `Offline model is missing`

For a pip installation, activate the virtual environment and run:

```console
formulaocr-offline --download-model
```

If the model is already elsewhere, pass `--model-dir` or set
`FORMULA_OCR_MODEL_DIR`.

### `Recognition dependencies are not installed`

The base Python project was installed without its runtime extra. From the
cloned project directory, run:

```console
python -m pip install '.[runtime]'
```

### NumPy cannot find `libstdc++.so.6` or another shared library

Read the last `Original error was:` line first. NumPy may follow a missing
shared-library error with a generic and misleading warning about importing from
its source directory.

On NixOS, deactivate the ordinary virtual environment and use either the Nix
package or the repository's devenv environment:

```console
deactivate 2>/dev/null || true
hash -r
nix run github:dragonleopardpig/formulaocr-offline -- formula.png

# For a source checkout instead:
devenv shell
formulaocr-offline --download-model
```

On Debian, Ubuntu, Fedora, or Arch Linux, install the native runtime packages
listed in the prerequisites above, then reactivate or recreate the virtual
environment. Avoid fixing this by globally setting `LD_LIBRARY_PATH`; it can
make unrelated programs load incompatible libraries.

### `No matching distribution found for paddlepaddle`

Confirm that Python is version 3.10 through 3.13 and that the platform is
supported by PaddlePaddle. Consult the
[official installation guide](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/paddlepaddle_installation.en.md)
for platform-specific CPU or GPU wheels.

### `The image does not look like a formula crop`

Crop the image more tightly and use a plain, high-contrast background. If the
input really is a formula and is still rejected, retry with `--no-classify`.

### `wl-copy is required for --copy`

Install `wl-clipboard`, omit `--copy`, or pipe standard output to the clipboard
utility used by the current desktop environment.

## Offline and security behavior

The pip installation needs network access while installing dependencies and
while explicitly running `--download-model`. The Nix build needs network access
to obtain its pinned source and model derivations. After those setup steps:

- recognition reads only the supplied local image and model files;
- the recognizer does not call a web API;
- a missing model produces an error instead of an automatic download;
- generated TeX is length-limited and checked for balanced braces/environments;
- TeX commands capable of reading files, writing files, or loading packages are
  rejected before output is returned.

The command recognizes LaTeX text; it does not execute or render that LaTeX.
Applications consuming the result must still render untrusted TeX safely.

## Model source and licensing

`PP-FormulaNet_plus-L` is developed by the PaddleOCR/PaddlePaddle team. Official
resources are:

- [PaddleOCR formula-recognition documentation](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/pipeline_usage/formula_recognition.en.md)
- [Official model card and files](https://huggingface.co/PaddlePaddle/PP-FormulaNet_plus-L)
- [Official inference-model archive](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-FormulaNet_plus-L_infer.tar)

Project code is MIT-licensed. PaddleOCR, PaddleX, PaddlePaddle, and their model
artifacts retain their respective upstream licenses. The model card currently
identifies the `PP-FormulaNet_plus-L` model license as Apache-2.0. Alternative
models and packages may use different licenses; users should review those terms
before downloading, modifying, or redistributing them.

## Development

The devenv shell includes the complete pip/uv runtime and the pinned model:

```console
devenv shell
python -m pytest
ruff check .
formulaocr-offline formula.png
```

The lighter Nix development shell is sufficient for unit tests and linting and
does not install the mutable pip runtime:

```console
nix develop
python -m pytest
ruff check .
nix flake check
```
