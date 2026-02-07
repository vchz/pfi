"""Top-level package for PFI estimators and utilities.

This module exposes the core subpackages for flow regression, score matching,
and utility helpers.
"""

import importlib as _importlib

__version__ = '0.1'

_submodules = [
    'flow',
    'score',
    'utils',
]

__all__ = _submodules + []


def __dir__():
    """List dynamic module attributes.

    Returns
    -------
    names : list of str
        Public names exposed by the package.
    """
    return __all__


def __getattr__(name):
    """Dynamically import known submodules on attribute access.

    Parameters
    ----------
    name : str
        Requested attribute name.

    Returns
    -------
    attr : module or object
        Imported submodule or existing global attribute.

    Raises
    ------
    AttributeError
        If ``name`` is not a known submodule or global symbol.
    """
    if name in _submodules:
        return _importlib.import_module(f"pfi.{name}")
    try:
        return globals()[name]
    except KeyError as exc:
        raise AttributeError(f"Module 'pfi' has no attribute '{name}'") from exc
