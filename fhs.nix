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
  name = "evox-env";

  targetPkgs = pkgs: with pkgs; [
    fish
    which
    swig
    stdenv
    gcc

    python310
    python310Packages.pip
    python310Packages.setuptools
    python310Packages.wheel
    python310Packages.virtualenv

    cudaPackages.cudatoolkit
    cudaPackages.cudnn
    cudaPackages.cutensor
    zlib
    SDL2
    glfw
  ];
}).env
