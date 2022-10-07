{ python3
, cmake
, gcc
, ninja
, git
, fetchFromGitHub
, zlib
, SDL2
, lib
}:
with python3.pkgs;
buildPythonPackage rec {
  pname = "ale-py";
  version = "0.8.0";
  format = "pyproject";
  rev-short = "d59d006";

  src = fetchFromGitHub {
    owner = "mgbellemare";
    repo = "Arcade-Learning-Environment";
    rev = "v${version}";
    sha256 = "OPAtCc2RapK1lALTKHd95bkigxcZ9bcONu32I/91HIg=";
  };

  buildInputs = [
    zlib
    SDL2
    setuptools
    wheel
    pybind11
  ];

  nativeBuildInputs = [
    cmake
    gcc
  ];

  propagatedBuildInputs = [
    typing-extensions
    importlib-resources
    numpy
  ] ++ lib.optionals (pythonOlder "3.10") [
    importlib-metadata
  ];

  checkInputs = [
    pytestCheckHook
    gym
  ];

  patches = [
    ./ale-py-cmake-pybind11.patch
  ];

  prePatch = ''
    substituteInPlace pyproject.toml \
      --replace 'dynamic = ["version"]' 'version = "${version}"'
    substituteInPlace setup.py \
      --replace 'subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=here)' 'b"${rev-short}"'
    cat setup.py
  '';

  buildPhase = ''
    mkdir -p dist
    ${python.interpreter} -m pip wheel --verbose --no-index --no-deps --no-clean --no-build-isolation --wheel-dir dist ../
  '';

  preCheck = ''
    cd ..
  '';

  doCheck = false;
}