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
  python = pkgs.python311;

  common-dependencies = with python.pkgs; [
    jax
    (if cudaSupport then jaxlibWithCuda else jaxlib)
    optax
    pyarrow
  ];

  doc-dependencies = with python.pkgs; [
    myst-parser
    numpydoc
    sphinx
    sphinx-book-theme
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
