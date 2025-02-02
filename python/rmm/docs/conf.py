# Copyright (c) 2020-2025, NVIDIA CORPORATION.

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
import datetime
import os
import re

from packaging.version import Version

import rmm

# -- Project information -----------------------------------------------------

project = "rmm"
copyright = f"2018-{datetime.datetime.today().year}, NVIDIA Corporation"
author = "NVIDIA Corporation"

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
RMM_VERSION = Version(rmm.__version__)
# The short X.Y version.
version = f"{RMM_VERSION.major:02}.{RMM_VERSION.minor:02}"
# The full version, including alpha/beta/rc tags.
release = (
    f"{RMM_VERSION.major:02}.{RMM_VERSION.minor:02}.{RMM_VERSION.micro:02}"
)


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

extensions = [
    "sphinxcontrib.jquery",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx_copybutton",
    "numpydoc",
    "sphinx_markdown_tables",
    "IPython.sphinxext.ipython_console_highlighting",
    "IPython.sphinxext.ipython_directive",
    "nbsphinx",
    "recommonmark",
    "breathe",
]

# Breathe Configuration
breathe_projects = {"librmm": "../../../doxygen/xml"}
breathe_default_project = "librmm"

copybutton_prompt_text = ">>> "

ipython_mplbackend = "str"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

# The master toctree document.
master_doc = "index"

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = "en"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = []

# List of warnings to suppress
suppress_warnings = []

# if the file deprecated.xml does not exist in the doxygen xml output,
# breathe will fail to build the docs, so we conditionally add
# "deprecated.rst" to the exclude_patterns list
if not os.path.exists(
    os.path.join(breathe_projects["librmm"], "deprecated.xml")
):
    exclude_patterns.append("librmm_docs/deprecated.rst")
    suppress_warnings.append("toc.excluded")

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False


# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#

html_theme = "sphinx_rtd_theme"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
# html_theme_options = {}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []


# -- Options for HTMLHelp output ------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "rmmdoc"


# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',
    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (
        master_doc,
        "rmm.tex",
        "RMM Documentation",
        "NVIDIA Corporation",
        "manual",
    )
]


# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, "rmm", "RMM Documentation", [author], 1)]


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "rmm",
        "RMM Documentation",
        author,
        "rmm",
        "One line description of project.",
        "Miscellaneous",
    )
]


# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numba": ("https://numba.readthedocs.io/en/stable", None),
    "cuda-python": (
        "https://nvidia.github.io/cuda-python/cuda-bindings/",
        None,
    ),
}

# Config numpydoc
numpydoc_show_inherited_class_members = True
numpydoc_class_members_toctree = False

autoclass_content = "init"

nitpick_ignore = [
    ("py:class", "size_t"),
    ("py:class", "void"),
]


def on_missing_reference(app, env, node, contnode):
    if (refid := node.get("refid")) is not None and "hpp" in refid:
        # We don't want to link to C++ header files directly from the
        # Sphinx docs, those are pages that doxygen automatically
        # generates. Adding those would clutter the Sphinx output.
        return contnode

    python_names_to_skip = [x for x in dir(int) if not x.startswith("__")]
    if (
        node["refdomain"] == "py"
        and (reftarget := node.get("reftarget")) is not None
        and any(toskip in reftarget for toskip in python_names_to_skip)
    ):
        return contnode

    cpp_names_to_skip = [
        # External names
        "cudaStream_t",
        "cudaStreamLegacy",
        "cudaStreamPerThread",
        "thrust",
        "spdlog",
        "stream_ref",
        # rapids_logger names
        "rapids_logger",
        # libcu++ names
        "cuda",
        "cuda::mr",
        "resource",
        "resource_ref",
        "async_resource",
        "async_resource_ref",
        "device_accessible",
        "host_accessible",
        "forward_property",
        "enable_if_t",
        # Unknown types
        "int64_t",
        "int8_t",
        # Internal objects
        "detail",
        "RMM_EXEC_CHECK_DISABLE",
        # Template types
        "Base",
    ]
    if (
        node["refdomain"] == "cpp"
        and (reftarget := node.get("reftarget")) is not None
    ):
        if any(toskip in reftarget for toskip in cpp_names_to_skip):
            return contnode

        # Strip template parameters and just use the base type.
        if match := re.search("(.*)<.*>", reftarget):
            reftarget = match.group(1)

        # This is the document we're linking _from_, and hence where
        # we should try and resolve the xref wrt.
        refdoc = node.get("refdoc")
        # Try to find the target prefixed with e.g. namespaces in case that's
        # all that's missing. Include the empty prefix in case we're searching
        # for a stripped template.
        extra_prefixes = ["rmm::", "rmm::mr::", "mr::", ""]
        for name, dispname, typ, docname, anchor, priority in env.domains[
            "cpp"
        ].get_objects():
            for prefix in extra_prefixes:
                if (
                    name == f"{prefix}{reftarget}"
                    or f"{prefix}{name}" == reftarget
                ):
                    return env.domains["cpp"].resolve_xref(
                        env,
                        refdoc,
                        app.builder,
                        node["reftype"],
                        name,
                        node,
                        contnode,
                    )

    return None


def setup(app):
    app.add_js_file("copybutton_pydocs.js")
    app.add_css_file("https://docs.rapids.ai/assets/css/custom.css")
    app.add_js_file(
        "https://docs.rapids.ai/assets/js/custom.js", loading_method="defer"
    )
    app.connect("missing-reference", on_missing_reference)
