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


# -- Project information -----------------------------------------------------

project = "EvoX"
copyright = "2022, Bill Huang"
author = "Bill Huang"


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

autodoc2_packages = [
    "../../src/evox",
]

autodoc2_render_plugin = "myst"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# mock these modules, so we can build the document without these dependencies.
autodoc_mock_imports = [
    "brax",
    "torchvision",
]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_js_files = ["https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.6/require.min.js"]
html_theme = "sphinx_book_theme"
html_logo = "_static/evox_logo_light.png"
html_theme_options = {
    "repository_branch": "main",
    "path_to_docs": "/docs/source",
    "repository_url": "https://github.com/EMI-Group/evox",
    "use_repository_button": True,
    "launch_buttons": {"colab_url": "https://colab.research.google.com"},
    "logo": {
        "image_light": "_static/evox_logo_light.png",
        "image_dark": "_static/evox_logo_dark.png",
        "alt_text": "EvoX logo image",
        "text": "",
    },
    "pygments_light_style": "default",
    "pygments_dark_style": "monokai",
}
favicons = [
    "favicon-16x16.ico",
    "favicon-32x32.ico",
]


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

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
myst_enable_extensions = ["dollarmath"]
locale_dirs = ["locale/"]
gettext_compact = "docs"
