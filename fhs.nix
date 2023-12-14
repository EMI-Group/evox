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

    python311
    python311Packages.pip
    python311Packages.setuptools
    python311Packages.wheel
    python311Packages.virtualenv

    cudaPackages.cudatoolkit
    cudaPackages.cudnn
    cudaPackages.cutensor
    zlib
    SDL2
    glfw
  ];
}).env
