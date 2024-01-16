"""
This script is used to wrap the translated strings in jupyter notebooks in the docs.po file
"""

import polib
import json

po = polib.pofile("docs/source/locale/zh/LC_MESSAGES/docs.po")


def is_from_notebook(occurrences):
    for filename, _linenum in occurrences:
        if filename.endswith(".ipynb"):
            return True

    return False


for entry in po:
    if is_from_notebook(entry.occurrences) and entry.msgstr != "":
        wrapped = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [entry.msgstr],
                },
            ],
            "metadata": {
                "kernelspec": {
                    "display_name": "venv",
                    "language": "python",
                    "name": "python3",
                },
                "language_info": {
                    "codemirror_mode": {"name": "ipython", "version": 3},
                    "file_extension": ".py",
                    "mimetype": "text/x-python",
                    "name": "python",
                    "nbconvert_exporter": "python",
                    "pygments_lexer": "ipython3",
                    "version": "3.10.12",
                },
            },
            "nbformat": 4,
            "nbformat_minor": 2,
        }
        entry.msgstr = json.dumps(wrapped)

po.save("docs/source/locale/zh/LC_MESSAGES/docs.po")
