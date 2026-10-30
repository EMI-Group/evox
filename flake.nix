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
        env = import ./dev_env_builder.nix { inherit system nixpkgs; };
      in {
        devShells.default = env;
        devShells.cpu = env;
      });
}
