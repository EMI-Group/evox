import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[0]
for _path in (str(_ROOT), str(_ROOT / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)
