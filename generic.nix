{ nixpkgs
, system
, cudaSupport
}:
let
  pkgs = import nixpkgs {
    inherit system;
    config.allowUnfree = true;
    config.cudaSupport = cudaSupport;
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

  # ray = import ./ray-bin.nix { inherit pkgs python; };
  # ray = local-nixpkgs.legacyPackages.x86_64-linux.python310Packages.ray-bin;

  dependencies = ps: with ps; [
    bokeh
    build
    chex
    flax
    gym
    jax
    jaxlib
    numpydoc
    optax
    pandas
    pydata-sphinx-theme
    pytest
    (ray-bin.override {
      withData = false;
      withServe = false;
      withTune = false;
      withRllib = false;
      withAir = false;
    })
    sphinx
    torchvision
    pytorch

    # gym render
    pyglet

    (pkgs.callPackage ./ale-py.nix {})
    (pkgs.callPackage ./auto-rom.nix {})
  ];


  pyenv = python.withPackages dependencies;

  evoxlib = python.pkgs.buildPythonPackage {
    pname = "evoxlib";
    version = "0.0.2";
    format = "pyproject";

    src = builtins.path { path = ./.; name = "evoxlib"; };
    propagatedBuildInputs = dependencies python.pkgs;
    checkInputs = [ python.pkgs.pytestCheckHook ];

    pythonImportsCheck = [
      "evoxlib"
    ];
    disabledTestPaths = [
      "tests/test_neuroevolution.py"
    ];
  };

  accelerator = if cudaSupport then "cuda" else "cpu";
in
{
  packages."${accelerator}" = evoxlib;

  devShells."${accelerator}" = pkgs.mkShell {
    buildInputs = [
      pyenv
    ];
  };
}
