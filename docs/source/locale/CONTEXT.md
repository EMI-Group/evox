# locale/ — Sphinx i18n Translation Directory
## Intent
This directory holds the **internationalization (i18n) resources** for the EvoX documentation, which is bilingual English ↔ Simplified Chinese (`zh_CN`).
It is wired into the build via `locale_dirs = ["locale/"]` and `gettext_compact = "docs"` in `conf.py`, and excluded from source scanning via `exclude_patterns = ["locale/**"]`.
The documentation content itself was rewritten for the experimental EvoX 2.0 pre-release (import name `evox`); the translation resources are being brought over to that content, and the gettext catalog still holds entries extracted from the 1.x pages.

## Dual translation strategy
| Content type | Mechanism | Location |
|---|---|---|
| **Tutorial pages** | Full-page manual translation (one Markdown file per English page) | `zh_CN/tutorial/` |
| **Everything else** (guide, examples, misc, API labels) | Gettext `.po` catalog | `zh_CN/LC_MESSAGES/docs.po` |

Full-page translation is used for the tutorial so the Chinese onboarding prose reads naturally and needs no gettext lookup.
At build time the `.readthedocs.yaml` pre-build job copies `zh_CN/tutorial/` over `docs/source/tutorial/` for non-English languages, so only the files that exist in `zh_CN/tutorial/` replace English pages — pages without a translation stay English.

## Current translation state (EvoX 2.0 pre-release)
- `zh_CN/tutorial/index.md` is translated and marked as work-in-progress; its toctree points at the English `tutorial_part1..7` pages until each part is retranslated.
- The 1.x Chinese translations of `tutorial_part1..7` were removed: they taught the removed torch API (`ModuleBase`, `Parameter`/`Mutable`, `torch.compile`/`vmap`) and would be actively misleading.
- `zh_CN/LC_MESSAGES/docs.po` is **stale**: its `msgid`s were extracted from the 1.x English pages, so most no longer match the 2.0 sources. Unmatched entries are simply unused by gettext (the English text is shown), so they never fail the build; orphaned entries (e.g. deleted `torch_and_evox.md`, deleted example notebooks, `guide/experimental/`) are harmless and are cleaned up on the next catalog refresh.

## Refreshing the catalog for a new or changed page
```bash
# 1) extract the current English strings into the POT file (gettext_compact = "docs")
PYTHONPATH=src python -m sphinx -b gettext docs/source docs/source/locale/gettext
# 2) merge them into the Chinese catalog (sphinx-intl is not in the docs extra — install it separately)
sphinx-intl update -p docs/source/locale/gettext -l zh_CN
# 3) translate the new/fuzzy entries, then run the maintenance scripts in docs/
python docs/fix_output.py
```
Translation of new pages is a manual (or reviewed) step — the repository does not machine-translate documentation as part of the build.

## Constraints
- `locale/**` must stay in `exclude_patterns`; the directory is only consumed through the i18n subsystem.
- Files in `zh_CN/tutorial/` must keep the exact filenames of their English counterparts for the `.readthedocs.yaml` copy step to work.
- The `.po` file is tracked in git, not a build artifact.
- Docs page text never depends on the locale; only `conf.py` nav labels read `READTHEDOCS_LANGUAGE`.
- Only `zh_CN` is supported. Adding a language needs a new locale subtree plus a language-code mapping in `.readthedocs.yaml` (a repo-root file, outside this node).

## Routing Table
| Area | Child Directory | Description |
|---|---|---|
| Chinese translations | `zh_CN/` | Simplified Chinese `.po` catalog and translated tutorial pages |
