# Contributing

## Documentation

Documentation is built with MkDocs - for full documentation visit [mkdocs.org](https://www.mkdocs.org). To install the packages needed for local development, run:
```
pip install graphreadability[docs]
```

This will install `mkdocs` (documentation generator), `mkdocs-material` (theme), `mkdocstrings` (plugin adding lots of cross-referencing abilities), and `mkdocstrings-python` (the additional handler needed for Python).

### MkDocs Commands

* `mkdocs serve` - Start the live-reloading docs server.
* `mkdocs build` - Build the documentation site.
* `mkdocs -h` - Print help message and exit.

### MkDocs Project layout

    mkdocs.yml    # The configuration file.
    docs/
        index.md  # The documentation homepage.
        ...       # Other markdown pages, images and other files.
