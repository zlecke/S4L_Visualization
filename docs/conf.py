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
sys.path.insert(0, os.path.abspath('..'))


# -- Project information -----------------------------------------------------

project = 'S4L Field Visualization'
copyright = '2021, Zach Eckert'
author = 'Zach Eckert'

# The full version, including alpha/beta/rc tags
release = '1.0.0'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx_rtd_theme',
              'numpydoc',
              'sphinx.ext.intersphinx',
              'sphinx.ext.autodoc']

# Intersphinx configuration
intersphinx_mapping = {'mayavi':     ('https://docs.enthought.com/mayavi/mayavi/', None),
                       'traitsui':   ('https://docs.enthought.com/traitsui/', None),
                       'traits':     ('https://docs.enthought.com/traits/', None),
                       'pyface':     ('https://docs.enthought.com/pyface/', None),
                       'matplotlib': ('https://matplotlib.org/stable/', None)}

# numpydoc configuration
numpydoc_show_class_members = False
numpydoc_validation_checks = {"all",
                              "SS06",
                              "RT01",
                              "RT02",
                              "RT03",
                              "RT04",
                              "RT05",
                              "YD01",
                              "SA01",
                              "EX01"}

# autodoc configuration
autodoc_default_options = {
        'member-order': 'bysource',
        'class-doc-from': 'both'
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

html_theme_options = {'collapse_navigation':        True,
                      'prev_next_buttons_location': 'both',
                      'navigation_depth':           -1}

add_module_names = False

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
