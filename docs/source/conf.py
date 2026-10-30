# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath("../../src"))
readthedocs_lang = os.environ.get("READTHEDOCS_LANGUAGE", "en")
readthedocs_ver = os.environ.get("READTHEDOCS_VERSION_NAME", "latest")
readthedocs_canonical = os.environ.get("READTHEDOCS_CANONICAL_URL", "")


# -- Project information -----------------------------------------------------

project = "EvoX"
copyright = "2022, Bill Huang"
author = "Bill Huang"
# The documented API is the experimental EvoX 2.0 pre-release (package
# ``evox``); the 1.x torch package ``evox`` no longer exists in this tree.
# Keep these in step with ``[project] version`` in ``pyproject.toml`` (2.0.0a1).
version = "2.0.0a1"
release = "2.0.0a1"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "autodoc2",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_favicon",
    "sphinxcontrib.mermaid",
    "myst_nb",
]

# The API documentation is generated from the functional EvoX 2.0 package ONLY
# (``src/evox``, built on the ``evoxc`` compiler). The 1.x torch package
# ``src/evox`` is not scanned: it was deleted in 2.0 and no longer exists.
autodoc2_packages = [
    "../../src/evox",
]

# ``autodoc2_module_all_regexes`` makes autodoc2 filter members through a
# module's ``__all__``; it errors out for any matching module that has no
# ``__all__``. In ``evox`` only these four packages define ``__all__`` in
# every module (the algorithm/operator/metric/vis_tools leaf modules do not), so
# the filter is restricted to them — everything else is documented by its public
# member set.
autodoc2_module_all_regexes = [
    r"evox\.core\..*",
    r"evox\.problems\..*",
    r"evox\.utils\..*",
    r"evox\.workflows\..*",
]

autodoc2_render_plugin = "myst"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "locale/**",
    # Agent-facing Context Tree files are not documentation pages.
    "CONTEXT.md",
    "**/CONTEXT.md",
    # autodoc2 documents this module (it is imported by ``evox.problems``)
    # but the parent package's ``__all__`` does not list it, so autodoc2 never
    # links it from a toctree. Excluding the generated page avoids an
    # "orphan document" warning; nothing links to it.
    "apidocs/evox/evox.problems.numerical.state.md",
]

# The documented package (``evox``) is pure Python on top of the ``evoxc``
# compiler; no third-party environment packages need to be mocked any more.
autodoc_mock_imports = []

# -- Options for HTML output -------------------------------------------------

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
html_baseurl = readthedocs_canonical

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
tutorial_title = "Tutorial" if readthedocs_lang == "en" else "教程"
api_title = "API Doc" if readthedocs_lang == "en" else "API文档"
example_title = "Examples" if readthedocs_lang == "en" else "示例"
ecosystem_title = "Ecosystem" if readthedocs_lang == "en" else "生态系统"

html_js_files = ["https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.7/require.min.js"]
html_theme = "shibuya"
html_logo = "_static/evox_logo_dark.svg"
html_context = {
    "languages": [
        ("English", f"/en/{readthedocs_ver}/%s.html", "en"),
        ("中文", f"/zh-cn/{readthedocs_ver}/%s.html", "zh-cn"),
    ],
    "source_type": "github",
    "source_user": "EMI-Group",
    "source_repo": "evox",
    "source_version": "main",
    "source_docs_path": "/docs/source/",
}
html_theme_options = {
    "toctree_maxdepth": 5,
    "light_logo": "_static/evox_brand_dark.svg",
    "dark_logo": "_static/evox_brand_light.svg",
    "og_image_url": "_static/evox_logo_dark.svg",
    "github_url": "https://github.com/EMI-Group/evox",
    "discord_url": "https://discord.gg/Vbtgcpy7G4",
    "nav_links": [
        {
            "title": tutorial_title,
            "url": "tutorial/index",
        },
        {
            "title": api_title,
            "url": "apidocs/index",
        },
        {
            "title": example_title,
            "url": "examples/index",
        },
        {
            "title": ecosystem_title,
            "url": "https://evox.group",
            "external": True,
            "children": [
                {
                    "title": "EvoMO",
                    "url": "https://github.com/EMI-Group/evomo",
                },
                {
                    "title": "EvoRL",
                    "url": "https://github.com/EMI-Group/evorl",
                },
                {
                    "title": "EvoGP",
                    "url": "https://github.com/EMI-Group/evogp",
                },
                {
                    "title": "TensorNEAT",
                    "url": "https://github.com/EMI-Group/tensorneat",
                },
                {
                    "title": "TensorRVEA",
                    "url": "https://github.com/EMI-Group/tensorrvea",
                },
                {"title": "TensorACO", "url": "https://github.com/EMI-Group/tensoraco"},
                {"title": "EvoXBench", "url": "https://github.com/EMI-Group/evoxbench"},
            ],
        },
    ],
    # ``source_*`` / ``github_url`` above already drive the shibuya repository
    # and edit-on-GitHub links.
}
favicons = [
    "favicon-16x16.ico",
    "favicon-32x32.ico",
]


autodoc_typehints_format = "short"
autodoc_typehints = "description"
autodoc_class_signature = "separated"
autosummary_generate = True
napoleon_google_docstring = False
napoleon_numpy_docstring = True
numpydoc_show_class_members = False
autosummary_generate = True
autosummary_imported_members = True
nb_execution_mode = "off"
nbsphinx_execute = "never"
nb_output_stderr = "remove"
myst_enable_extensions = [
    "dollarmath",
    "fieldlist",
    "linkify",
    "colon_fence",
    "deflist",
    "attrs_inline",
]
locale_dirs = ["locale/"]
gettext_compact = "docs"
