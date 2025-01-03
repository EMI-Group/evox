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
  pythonPackages = pkgs.python3Packages;
in
pkgs.mkShell rec {
  name = "impureEvoXPythonEnv";
  venvDir = "./.venv";
  buildInputs = with pythonPackages; [
    python
    # This executes some shell code to initialize a venv in $venvDir before
    # dropping into the shell
    venvShellHook

    # Those are dependencies that we would like to use from nixpkgs, which will
    # add them to PYTHONPATH and thus make them accessible from within the venv.
    numpy
    torch
  ] ++ (with pkgs; [
    pre-commit
    ruff
  ]);

  # Run this command, only after creating the virtual environment
  postVenvCreation = ''
    unset SOURCE_DATE_EPOCH
    pip install -e .
  '';

  # Now we can execute any commands within the virtual environment.
  # This is optional and can be left out to run pip manually.
  postShellHook = ''
    # allow pip to install wheels
    unset SOURCE_DATE_EPOCH
  '';

}

