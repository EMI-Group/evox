{ nixpkgs, system, cudaSupport ? false, rocmSupport ? false
, vulkanSupport ? false }:
let
  pkgs = import nixpkgs {
    inherit system;
    config.allowUnfree = true;
    config.cudaSupport = cudaSupport;
    config.rocmSupport = rocmSupport;
    config.vulkanSupport = vulkanSupport;
  };
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
        (if cudaSupport then
          torchWithCuda
        else if rocmSupport then
          torchWithRocm
        else if vulkanSupport then
          torchWithVulkan
        else
          torch)
        torchvision
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
  '';
}
