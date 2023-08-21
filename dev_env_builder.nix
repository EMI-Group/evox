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
    pyarrow
  ];

  doc-dependencies = with python.pkgs; [
    flax
    myst-parser
    numpydoc
    pydata-sphinx-theme
    sphinx
    sphinx-copybutton
    sphinx-design
  ];

  test-dependencies = with python.pkgs; [
    pytest
    chex
    gym
    ray
    torchvision
  ];
in
pkgs.mkShell {
  buildInputs = [
    (python.withPackages
      (ps: common-dependencies
        ++ test-dependencies
        ++ doc-dependencies))
  ];
}
