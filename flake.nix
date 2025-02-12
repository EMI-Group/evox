{
  description = "evox";
  inputs = {
    nixpkgs.url = "nixpkgs/nixos-unstable";
    utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, utils }:
    with utils.lib;
    with nixpkgs.lib;
    eachSystem (with system; [ x86_64-linux ]) (system:
      let
        builder = import ./dev_env_builder.nix;
        cuda-env = builder {
          inherit system nixpkgs;
          cudaSupport = true;
        };
        rocm-env = builder {
          inherit system nixpkgs;
          rocmSupport = true;
        };
        vulkan-env = builder {
          inherit system nixpkgs;
          vulkanSupport = true;
        };
        cpu-env = builder { inherit system nixpkgs; };
      in {
        devShells.default = cpu-env;
        devShells.cpu = cpu-env;
        devShells.cuda = cuda-env;
        devShells.rocm = rocm-env;
        devShells.vulkan = vulkan-env;
      });
}
