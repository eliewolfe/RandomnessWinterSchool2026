"""Helpers for importing the vendored `inflation` submodule.

The SDP backend will use the devel branch of `inflation` only for
canonicalization / commutation utilities. This module keeps the import-path
logic in one place so the rest of the codebase can remain clean.
"""

from __future__ import annotations

from importlib import util as importlib_util
from pathlib import Path
import sys


def inflation_submodule_root() -> Path:
    """Return the checked-out root directory of the `inflation` submodule."""
    return Path(__file__).resolve().parents[1] / "external" / "inflation"


def ensure_inflation_importable() -> Path:
    """Add the inflation submodule root to `sys.path` if it is available.

    Returns the resolved submodule root path. Raises `FileNotFoundError` if the
    submodule has not been checked out yet.
    """
    root = inflation_submodule_root()
    if not root.is_dir():
        raise FileNotFoundError(
            f"inflation submodule is missing at {root}. Run `git submodule update --init --recursive`."
        )
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def import_inflation() -> object:
    """Import and return the vendored `inflation` package."""
    ensure_inflation_importable()
    import inflation  # type: ignore[import-not-found]

    return inflation


def load_inflation_source_module(relative_path: str, *, module_name: str | None = None) -> object:
    """Load a Python source file from the inflation submodule without importing `inflation/__init__.py`.

    This is useful because the submodule's package initializer eagerly imports
    solver-facing components that we do not want to pull in just to access
    canonicalization helpers.
    """
    root = ensure_inflation_importable()
    module_path = root / relative_path
    if not module_path.is_file():
        raise FileNotFoundError(f"Inflation source file not found: {module_path}")

    name = module_name or f"inflation_source_{relative_path.replace('/', '_').replace('\\\\', '_').replace('.py', '')}"
    spec = importlib_util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to build an import spec for {module_path}")
    module = importlib_util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_fast_npa_module() -> object:
    """Load `inflation/sdp/fast_npa.py` directly from the submodule checkout."""
    return load_inflation_source_module("inflation/sdp/fast_npa.py", module_name="inflation_fast_npa")


