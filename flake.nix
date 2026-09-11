{
  description = "Fully offline formula-image to LaTeX recognition";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs =
    { self, nixpkgs }:
    let
      system = "x86_64-linux";
      overlay = final: _previous: {
        formulaocr-offline = final.callPackage ./nix/package.nix { source = self; };
      };
      pkgs = import nixpkgs {
        inherit system;
        overlays = [ overlay ];
      };
      package = pkgs.formulaocr-offline;
      testPython = pkgs.python313.withPackages (
        pythonPackages: with pythonPackages; [
          numpy
          pillow
          pytest
        ]
      );
    in
    {
      overlays.default = overlay;

      packages.${system} = {
        default = package;
        formulaocr-offline = package;
      };

      apps.${system}.default = {
        type = "app";
        program = "${package}/bin/formulaocr-offline";
        meta = package.meta;
      };

      checks.${system} = {
        inherit package;
        unit-tests =
          pkgs.runCommand "formulaocr-offline-unit-tests"
            {
              nativeBuildInputs = [ testPython ];
            }
            ''
                  cp -r ${self} source
                  chmod -R u+w source
                  cd source
              python -m pytest -q
                  touch $out
            '';
      };

      devShells.${system}.default = pkgs.mkShell {
        packages = [
          pkgs.nixfmt
          pkgs.ruff
          testPython
        ];
      };

      formatter.${system} = pkgs.nixfmt;

      nixosModules.default =
        { lib, pkgs, ... }:
        {
          nixpkgs.overlays = [ overlay ];
          environment.systemPackages = [ pkgs.formulaocr-offline ];
          environment.variables.MINDER_FORMULA_OCR = lib.getExe pkgs.formulaocr-offline;
        };
    };
}
