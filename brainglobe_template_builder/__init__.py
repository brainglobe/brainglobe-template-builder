from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("brainglobe-template-builder")
except PackageNotFoundError:
    # package is not installed
    pass
