# _static/ — Static site assets
## Intent
This directory holds the static assets of the EvoX documentation: the brand logos, the favicons and the theme CSS that the built site copies verbatim into `_build/html/_static/`.
It stores assets only — no documentation pages live here, and nothing here describes an EvoX API, so the 2.0-only rule applies indirectly: an asset that advertises or depicts the removed 1.x torch API does not belong in the tree.
## API Surface
| File | Purpose | Referenced from |
|---|---|---|
| `evox_logo_dark.svg` | Dark-background EvoX logo; also the Open Graph/social-preview image | `conf.py` (`html_logo`, theme `og_image_url`) and `guide/developer/document.md` |
| `evox_logo_light.svg` | Light-background counterpart of `evox_logo_dark.svg` | Currently unreferenced — kept as the other half of the brand logo pair |
| `evox_brand_dark.svg` | Brand wordmark used as the logo on the light theme | `conf.py` (theme option `light_logo`) |
| `evox_brand_light.svg` | Brand wordmark used as the logo on the dark theme | `conf.py` (theme option `dark_logo`) |
| `favicon-16x16.ico`, `favicon-32x32.ico` | Browser favicons | `conf.py` (`sphinx_favicon` `favicons` list) |
| `qq_group_number.jpg` | QR code of the EvoX community QQ group (297969717) | Currently unreferenced — a community asset, not a 1.x API asset, kept for a future community/contact page |
| `css/custom.css` | Two rules that make notebook cell inputs/outputs transparent | `conf.py` (`html_css_files`) |
## Constraints
- **Brand assets stay even when unreferenced**: logos, favicons, icons and theme images are configuration-driven (`conf.py`, shibuya theme options) or kept for reuse, so an unused brand asset is never a reason to delete it.
- **No 1.x-era assets**: the 1.x torch API is gone from this tree, so torch-era screenshots (`ModuleBase`, `jit_vmap`, `halfcheetah_200`, example run outputs) and the 1.x Windows one-click installer `win-install.bat` were deleted rather than archived; they advertised `pip install "evox[vis]>=1.0.0"`, a torch-CPU fallback and `triton-windows` for `torch.compile`, none of which exists in 2.0.
- **New screenshots must come from the 2.0 API**: a run output, diagram or installer script added here has to be produced from `evox` on the `evoxc` backends, and the page that embeds it must reference it by its `/_static/<name>` path.
- Every added asset has to be referenced by a page or by `conf.py`; an orphan that is also 1.x-era is deleted instead of being left in the tree.
- This directory is `html_static_path` in `conf.py`, so `CONTEXT.md` files are excluded from the build through `exclude_patterns` (`**/CONTEXT.md`) and are not copied into the output.
## Routing Table
| Child Directory | Description |
|---|---|
| `css/` | Theme CSS overrides loaded through `html_css_files` |
