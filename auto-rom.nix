{ python3
, bash
}:
with python3.pkgs;
buildPythonApplication rec {
  pname = "AutoROM";
  version = "0.4.2";
  format = "pyproject";

  src = fetchPypi {
    inherit pname version;
    sha256 = "b426a39bc0ee3781c7791f28963a9b2e4385b6421eeaf2f368edc00c761d428a";
  };

  buildInputs = [
    setuptools
    wheel
    build
  ];

  propagatedBuildInputs = [
    click
    requests
    tqdm
  ];

  checkPhase = ''
    AutoROM
  '';
}