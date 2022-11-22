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
    ray
    sphinx
    torchvision
    pytorch

    # gym render
    pyglet
  ];

  pyenv = python.withPackages dependencies;

  evox = python.pkgs.buildPythonPackage {
    pname = "evox";
    version = "0.0.2";
    format = "pyproject";

    src = builtins.path { path = ./.; name = "evox"; };
    propagatedBuildInputs = dependencies python.pkgs;
    checkInputs = [ python.pkgs.pytestCheckHook ];

    pythonImportsCheck = [
      "evox"
    ];
    disabledTestPaths = [
      "tests/test_neuroevolution.py"
    ];
  };

  accelerator = if cudaSupport then "cuda" else "cpu";
in
{
  packages."${accelerator}" = evox;

  devShells."${accelerator}" = pkgs.mkShell {
    buildInputs = [
      pyenv
    ];
  };
}
