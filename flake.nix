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
    utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in
        with pkgs;
        rec {
          python = python310;

          pyenv = python.withPackages (ps: with ps; [
            pytest
            chex
            jax
            jaxlib
          ]);

          devShell = mkShell {
            buildInputs = [
              pyenv
            ];
          };
        }
    );
}
