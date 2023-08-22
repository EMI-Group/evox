{
  description = "evox";
  inputs = {
    nixpkgs.url = "nixpkgs/nixos-unstable";
    utils.url = "github:numtide/flake-utils";
    flake-compat = {
      url = "github:edolstra/flake-compat";
      flake = false;
    };
  };

  outputs = { self, nixpkgs, utils, flake-compat }:
    with utils.lib;
    with nixpkgs.lib;
    eachSystem (with system; [ x86_64-linux ]) (system:
      let
        builder = import ./dev_env_builder.nix;
        cuda-env = builder { inherit system nixpkgs; cudaSupport = true; };
        cpu-env = builder { inherit system nixpkgs; cudaSupport = false; };
      in
      {
        devShells.default = cpu-env;
        devShells.cpu = cpu-env;
        devShells.cuda = cuda-env;
        devShells.fhs = import ./fhs.nix {
          inherit nixpkgs system; cudaSupport=true;
        };
      }
    );
}
