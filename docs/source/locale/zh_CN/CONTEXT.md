# locale/zh_CN — Simplified Chinese Localization
## Intent
This directory holds all **Simplified Chinese** resources for the EvoX documentation: the gettext catalog used for the guides, examples and API labels, and the full-page translations of the tutorial.
It is the only non-English locale; `READTHEDOCS_LANGUAGE` (read in `conf.py`, default `en`) selects it.

## API Surface
```
zh_CN/
├── LC_MESSAGES/
│   └── docs.po          ← gettext catalog (msgid English → msgstr Chinese)
└── tutorial/
    └── index.md         ← translated tutorial landing page (work in progress)
```

### `LC_MESSAGES/docs.po`
- Standard GNU gettext catalog, wired through `locale_dirs = ["locale/"]` + `gettext_compact = "docs"`.
- **Stale relative to the EvoX 2.0 pre-release content**: its `msgid`s were extracted from the 1.x pages.
  Entries that no longer match any English string are ignored by gettext (English is shown) and cannot fail the build.
- Entries referencing deleted pages (`torch_and_evox.md`, the former example notebooks, `guide/experimental/`) are orphans to be dropped at the next catalog refresh.

### `tutorial/`
- `index.md` is translated and declares the tutorial's translation status; its toctree points at the English `tutorial_part1..7` pages.
- The 1.x translations of `tutorial_part1..7` were deleted because they documented the removed torch API.
- Files here must keep the exact filenames of their English counterparts: the `.readthedocs.yaml` pre-build job runs `cp -r docs/source/locale/zh_CN/tutorial docs/source`, so a page with no translated file simply stays English.

## Relationship to the English source
| English source | zh_CN counterpart |
|---|---|
| `docs/source/tutorial/*.md` | `docs/source/locale/zh_CN/tutorial/*.md` (only translated parts) |
| all other pages + API labels | entries in `LC_MESSAGES/docs.po` |

## Constraints
- Never machine-translate the tutorial `.md` files: they are hand-written translations.
- Refresh the catalog with `sphinx-build -b gettext` + `sphinx-intl update -p <pot-dir> -l zh_CN`, then translate by hand and run `docs/fix_output.py`.
- Do not add pages here that have no English counterpart — the copy step would publish them under the English URL tree.
- `docs.po` is tracked in git; translation is also coordinated through Weblate (see the `Language-Team` header of the catalog).

## Routing Table
Leaf node — no further subdirectories.
