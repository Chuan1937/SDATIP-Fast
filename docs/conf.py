import os
import sys
import subprocess
from datetime import datetime

# Include docs directory
sys.path.insert(0, os.path.abspath(".."))

project = "SDATIP-Fast"
copyright = f"{datetime.now().year}, He XingChen"
author = "He XingChen"

def get_git_last_updated(file_path):
    """Get the last modified date of a file from git."""
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%ci", "--", file_path],
            cwd=os.path.abspath(".."),
            capture_output=True,
            text=True,
            check=True,
        )
        if result.stdout.strip():
            git_date = datetime.strptime(result.stdout.strip(), "%Y-%m-%d %H:%M:%S %z")
            return git_date.strftime("%Y-%m-%d")
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        pass
    return None

def setup(app):
    """Setup function to add custom configuration."""
    def html_page_context_handler(app, pagename, templatename, context, doctree):
        if hasattr(app.config, 'source_suffix'):
            source_suffix = app.config.source_suffix
            if not isinstance(source_suffix, list):
                source_suffix = list(source_suffix.keys())

            source_file = None
            for suffix in source_suffix:
                possible_path = os.path.join(app.srcdir, pagename + suffix)
                if os.path.exists(possible_path):
                    source_file = possible_path
                    break

            if source_file:
                last_updated_str = get_git_last_updated(source_file)
                if last_updated_str:
                    context["last_updated"] = last_updated_str

        context["author"] = app.config.author
    app.connect("html-page-context", html_page_context_handler)

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

source_suffix = {
    ".md": "markdown",
    ".rst": "restructuredtext",
}

templates_path = ["_templates"]
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_last_updated_fmt = "%Y-%m-%d"