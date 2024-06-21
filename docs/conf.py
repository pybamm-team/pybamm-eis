# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

# Path for repository root
sys.path.insert(0, os.path.abspath("../"))

# -- Project information -----------------------------------------------------

project = "PyBaMM EIS"
copyright = "2022-2024, University of Oxford"
author = "The PyBaMM Team, Rishit Dhoot"

# -- General configuration ---------------------------------------------------

extensions = [
    # Sphinx extensions
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    # Third-party extensions
    "myst_parser",
]
myst_enable_extensions = [
    "dollarmath",
]
myst_dmath_double_inline = True
templates_path = ["_templates"]
source_suffix = [".rst", ".md"]
master_doc = "index"
language = "en"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

html_theme = "bizstyle"
html_static_path = ["source/_static"]
html_sidebars = {
    "**": ["globaltoc.html", "relations.html", "sourcelink.html", "searchbox.html"]
}
html_css_files = ["theme.css"]
