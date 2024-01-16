"""
This script is used to wrap the translated strings in jupyter notebooks in the docs.po file
"""

import polib
import json
import copy

po = polib.pofile("docs/source/locale/zh/LC_MESSAGES/docs.po")


def split_msg_from_notebook(occurrences):
    normal = []
    notebook = []
    for filename, linenum in occurrences:
        if filename.endswith(".ipynb"):
            notebook.append((filename, linenum))
        else:
            normal.append((filename, linenum))

    return normal, notebook


normal_entries = []
for entry in po:
    from_normal, from_notebook = split_msg_from_notebook(entry.occurrences)
    if from_notebook and entry.msgstr != "":
        original_msgstr = entry.msgstr
        entry.occurrences = from_notebook
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
        entry.msgctxt = "from notebook file"
        entry.msgstr = json.dumps(wrapped)

        if from_normal:
            normal_entry = copy.copy(entry)
            normal_entry.msgstr = original_msgstr
            normal_entry.occurrences = from_normal
            normal_entry.msgctxt = "from normal file"
            normal_entries.append(normal_entry)

for entry in normal_entries:
    po.append(entry)

po.save("docs/source/locale/zh/LC_MESSAGES/docs.po")
