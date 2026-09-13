{ pkgs, ... }:

let
  model = pkgs.fetchzip {
    url = "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-FormulaNet_plus-L_infer.tar";
    hash = "sha256-h19i49d4APTPuJtsF8TqqU1Ktp4HkdOkqrL7Pq/akEA=";
    stripRoot = true;
  };
in
{
  packages = with pkgs; [
    ccache
    wl-clipboard
  ];

  languages.python = {
    enable = true;
    package = pkgs.python312;
    libraries = with pkgs; [
      glib
      libglvnd
      zlib
    ];
    manylinux.enable = true;
    venv.enable = true;
    uv = {
      enable = true;
      sync = {
        enable = true;
        allExtras = true;
      };
    };
  };

  env = {
    FORMULA_OCR_MODEL_DIR = model;
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK = "True";
  };
}
