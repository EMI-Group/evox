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

        pydata-sphinx-theme = python.pkgs.buildPythonPackage rec {
          pname = "pydata_sphinx_theme";
          version = "0.9.0";
          format = "wheel";

          src = python.pkgs.fetchPypi {
            inherit pname version format;
            python = "py3";
            sha256 = "b22b442a6d6437e5eaf0a1f057169ffcb31eaa9f10be7d5481a125e735c71c12";
          };
          propagatedBuildInputs = with python.pkgs; [
            sphinx
            beautifulsoup4
            docutils
            packaging
            pygments
          ];
        };

        common-dependencies = ps: with ps; [
          build
          chex
          jax
          pytest
          sphinx
          pydata-sphinx-theme
          numpydoc
          bokeh
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

          devShells.default = devShells.cpu;
        }
    );
}
