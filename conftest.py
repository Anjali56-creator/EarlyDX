"""Ensure the repository root is importable as the top-level package path."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
