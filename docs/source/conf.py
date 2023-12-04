"""Configuration file for the Sphinx documentation builder.

For the full list of built-in configuration values, see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

import os
import sys
import pathlib

# sys.path.insert(0, os.path.abspath("../.."))
CURRENT_PATH = pathlib.Path(os.path.abspath(os.path.dirname(__file__)))
SKRM_PATH = CURRENT_PATH.parent.parent

sys.path.insert(0, str(SKRM_PATH))
import skreducedmodel

# GWTOOLS_PATH = /home/fcerino/env_skrm/lib/python3.10/site-packages
# sys.path.insert(0, GWTOOLS_PATH)

# import gwtools

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "ScikitReduceModel"
copyright = "2022, Franco Cerino - Agustín Rodríguez-Medrano"
author = "Franco Cerino - Agustín Rodríguez-Medrano"
release = "0.0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "nbsphinx",
]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'sphinx_pdj_theme'
# html_theme_path = [sphinx_pdj_theme.get_html_theme_path()]
html_theme = "sphinx_rtd_theme"
# html_theme = 'alabaster'
# html_static_path = ['_static']
