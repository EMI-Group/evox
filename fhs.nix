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
in
(pkgs.buildFHSUserEnv {
  name = "evoxlib-env";

  targetPkgs = pkgs: with pkgs; [
    fish
    which
    python39
    python39Packages.pip
    python39Packages.setuptools
    python39Packages.wheel
    python39Packages.virtualenv

    cudaPackages.cudatoolkit
    cudaPackages.cudnn
    cudaPackages.cutensor
    zlib
    SDL2
  ];

  runScript = "fish";
}).env
