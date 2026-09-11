{
  ccache,
  fetchzip,
  lib,
  makeWrapper,
  python313,
  source,
  stdenvNoCC,
  wl-clipboard,
}:

let
  offlinePython = python313.override {
    packageOverrides = _final: previous: {
      paddlex = previous.paddlex.overridePythonAttrs (old: {
        dontCheckRuntimeDeps = true;
        dependencies = builtins.filter (
          dependency:
          !(builtins.elem (lib.getName dependency) [
            "aistudio-sdk"
            "modelscope"
          ])
        ) old.dependencies;
        patches = (old.patches or [ ]) ++ [ ./paddlex-no-aistudio.patch ];
      });
    };
  };

  pythonEnvironment = offlinePython.withPackages (
    pythonPackages: with pythonPackages; [
      ftfy
      numpy
      opencv-contrib-python
      paddleocr
      paddlepaddle
      pillow
      pypdfium2
      tokenizers
    ]
  );

  model = fetchzip {
    url = "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-FormulaNet_plus-L_infer.tar";
    hash = "sha256-h19i49d4APTPuJtsF8TqqU1Ktp4HkdOkqrL7Pq/akEA=";
    stripRoot = true;
  };
in
stdenvNoCC.mkDerivation {
  pname = "formulaocr-offline";
  version = "0.1.0";
  src = source;

  nativeBuildInputs = [ makeWrapper ];

  dontBuild = true;

  installPhase = ''
    runHook preInstall

    mkdir -p $out/lib/formulaocr-offline
    cp -r formulaocr_offline $out/lib/formulaocr-offline/

    makeWrapper ${pythonEnvironment}/bin/python $out/bin/formulaocr-offline \
      --add-flags "-m formulaocr_offline" \
      --prefix PATH : ${
        lib.makeBinPath [
          ccache
          wl-clipboard
        ]
      } \
      --prefix PYTHONPATH : "$out/lib/formulaocr-offline" \
      --set FORMULA_OCR_MODEL_DIR ${model} \
      --set PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK True \
      --set PYTHONNOUSERSITE 1

    runHook postInstall
  '';

  doInstallCheck = true;
  installCheckPhase = ''
    test -f ${model}/inference.json
    test -f ${model}/inference.pdiparams
    test -f ${model}/inference.yml
    $out/bin/formulaocr-offline --help >/dev/null
  '';

  passthru = { inherit model; };

  meta = {
    description = "Offline PP-FormulaNet image-to-LaTeX recognizer";
    homepage = "https://github.com/dragonleopardpig/formulaocr-offline";
    license = with lib.licenses; [
      asl20
      mit
    ];
    mainProgram = "formulaocr-offline";
    platforms = [ "x86_64-linux" ];
  };
}
