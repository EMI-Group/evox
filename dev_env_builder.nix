{ nixpkgs, system }:
let
  pkgs = import nixpkgs { inherit system; };
in pkgs.mkShell {
  name = "impureEvoXPythonEnv";
  venvDir = "./.venv";
  nativeBuildInputs = with pkgs; [
    (python313.withPackages (py-pkgs:
      with py-pkgs; [
        # This executes some shell code to initialize a venv in $venvDir before
        # dropping into the shell
        venvShellHook
        # Those are dependencies that we would like to use from nixpkgs, which will
        # add them to PYTHONPATH and thus make them accessible from within the venv.
        numpy
        pytest
      ]))
    pre-commit
    ruff
  ];
  # Run this command, only after creating the virtual environment
  postVenvCreation = ''
    unset SOURCE_DATE_EPOCH
  '';
  # Now we can execute any commands within the virtual environment.
  # This is optional and can be left out to run pip manually.
  postShellHook = ''
    # allow pip to install wheels
    unset SOURCE_DATE_EPOCH
    # `evoxc` (the EvoX Compiler) is not packaged in nixpkgs; install it into the
    # venv from PyPI once per environment, e.g.
    #   pip install "evoxc>=0.1.0"                 # pure-numpy backend
    #   pip install "evoxc[compilers]>=0.1.0"      # + IREE / TVM / XLA backends
  '';
}
