{
  description = "Evoxlib";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    utils.url = "github:numtide/flake-utils";
    flake-compat = {
      url = "github:edolstra/flake-compat";
      flake = false;
    };
  };

  outputs = { self, nixpkgs, utils, flake-compat }:
    utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        python = pkgs.python310;
        dependencies = ps: with ps; [
          build
          chex
          jax
          jaxlib
          pytest
        ];
        pyenv = python.withPackages dependencies;
      in
        with pkgs; rec {
          packages.default = python.pkgs.buildPythonPackage rec {
            pname = "evoxlib";
            version = "0.0.1";
            format = "pyproject";

            src = ./.;
            propagatedBuildInputs = dependencies python.pkgs;

            checkPhase = ''
              python -m pytest
            '';
          };

          devShells.default = mkShell {
            buildInputs = [
              pyenv
            ];
          };

          check = packages.default;
        }
    );
}
