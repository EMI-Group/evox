{
  description = "Evoxlib";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    utils.url = "github:numtide/flake-utils";
    flake-compat = {
      url = "github:edolstra/flake-compat";
      flake = false;
    };
    local-nixpkgs.url = "/home/bill/Source/nixpkgs";
  };

  outputs = { self, nixpkgs, utils, flake-compat, local-nixpkgs }:
    with utils.lib;
    with nixpkgs.lib;
    eachSystem (with system; [ x86_64-linux ]) (system:
      let
        callPackage = nixpkgs.legacyPackages.${system}.callPackage;
        generic-builder = import ./generic.nix;
        with-cuda = generic-builder { inherit system; nixpkgs = local-nixpkgs; cudaSupport = true; };
        cpu-only = generic-builder { inherit system; nixpkgs = local-nixpkgs; cudaSupport = false; };
        total = recursiveUpdate with-cuda cpu-only;
      in
      recursiveUpdate total {
        devShells.default = total.devShells.cpu;
        packages.default = total.packages.cpu;
        packages.auto-rom = callPackage ./auto-rom.nix {};
      }
    );
}
