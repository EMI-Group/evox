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

  common-dependencies = with python.pkgs; [
    jax
    jaxlib
    optax
    bokeh
  ];

  doc-dependencies = with python.pkgs; [
    myst-parser
    numpydoc
    pydata-sphinx-theme
    sphinx
    sphinx-copybutton
    sphinx-design
  ];

  test-dependencies = with python.pkgs; [
    chex
    flax
    pytest
  ];

  other-dependencies = with python.pkgs; [
    gym
    ray
    torchvision
  ];

  evox = python.pkgs.buildPythonPackage {
    pname = "evox";
    version = "0.0.3";
    format = "pyproject";

    src = builtins.path { path = ./.; name = "evox"; };
    propagatedBuildInputs = common-dependencies;
    checkInputs = [ python.pkgs.pytestCheckHook ] ++ test-dependencies;

    pythonImportsCheck = [
      "evox"
    ];
  };

  accelerator = if cudaSupport then "cuda" else "cpu";
in
{
  packages."${accelerator}" = evox;

  devShells."${accelerator}" = pkgs.mkShell {
    buildInputs = [
      (python.withPackages
        (ps: common-dependencies
          ++ test-dependencies
          ++ doc-dependencies
          ++ other-dependencies))
    ];
  };
}
