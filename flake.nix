{
  description = "Evoxlib";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
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
        callPackage = nixpkgs.legacyPackages.${system}.callPackage;
        generic-builder = import ./generic.nix;

        with-cuda = generic-builder { inherit nixpkgs system; cudaSupport = true; };
        cpu-only = generic-builder { inherit nixpkgs system; cudaSupport = false; };
        total = recursiveUpdate with-cuda cpu-only;
      in
      recursiveUpdate total {
        devShells.default = total.devShells.cpu;
        packages.default = total.packages.cpu;
      }
    );
}
