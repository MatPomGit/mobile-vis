"""Project-local Python startup customizations.

Ensures that the ``src/`` directory is importable when commands are run
from the repository root without installing the package first.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC_PATH = Path(__file__).resolve().parent / "src"

if SRC_PATH.is_dir():
    src_path_str = str(SRC_PATH)
    if src_path_str not in sys.path:
        sys.path.insert(0, src_path_str)
