try:
    from wildcode._version import __version__, __version_tuple__
except ImportError:
    __version__ = "local-dev"
