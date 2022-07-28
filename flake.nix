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
    with utils.lib;
    eachSystem (with system; [ x86_64-linux ]) (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };
        python = pkgs.python310;
        common-dependencies = ps: with ps; [
          build
          chex
          jax
          pytest
        ];
        other-dependencies = gpuSupport: ps: with ps;
          if gpuSupport
          then [jaxlibWithCuda]
          else [jaxlibWithoutCuda];
        dependencies = gpuSupport: ps: common-dependencies ps ++ other-dependencies gpuSupport ps;

        cpu-pyenv = python.withPackages (dependencies false);
        gpu-pyenv = python.withPackages (dependencies true);

        evoxlib = gpuSupport: python.pkgs.buildPythonPackage {
          pname = "evoxlib";
          version = "0.0.1";
          format = "pyproject";

          src = builtins.path { path = ./.; name = "evoxlib"; };
          propagatedBuildInputs = dependencies gpuSupport python.pkgs;

          checkPhase = ''
            python -m pytest
          '';
        };
      in
        with pkgs; rec {
          packages.cpu = evoxlib false;
          packages.gpu = evoxlib true;
          packages.default = packages.gpu;

          devShells.cpu = mkShell {
            buildInputs = [
              cpu-pyenv
            ];
          };

          devShells.gpu = mkShell {
            buildInputs = [
              gpu-pyenv
            ];
          };

          devShells.default = devShells.gpu;
        }
    );
}
