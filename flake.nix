{
  description = "evox";
  inputs = {
    nixpkgs.url = "nixpkgs/nixos-24.11";
    utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, utils }:
    with utils.lib;
    with nixpkgs.lib;
    eachSystem (with system; [ x86_64-linux ]) (system:
      let
        builder = import ./dev_env_builder.nix;
        cuda-env = builder { inherit system nixpkgs; cudaSupport = true; rocmSupport = false; };
        rocm-env = builder { inherit system nixpkgs; cudaSupport = false; rocmSupport = true; };
        cpu-env = builder { inherit system nixpkgs; cudaSupport = false; rocmSupport = false; };
      in
      {
        devShells.default = cpu-env;
        devShells.cpu = cpu-env;
        devShells.cuda = cuda-env;
        devShells.rocm = rocm-env;
      }
    );
}
