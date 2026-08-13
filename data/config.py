"""
Data configuration and paths.

The Aalto dataset is stored externally to avoid duplication.
This module provides the paths to access it.
"""

from pathlib import Path

# Aalto dataset location (168,595 keystroke files)
AALTO_RAW_PATH = Path(
    r"C:\Users\andre\OneDrive\Personal Projects\Keystroke Biometrics\Keystrokes\Keystrokes\files"
)

# Project data directories
PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
PREPROCESSED_PATH = DATA_ROOT / "preprocessed"

# Ensure preprocessed directory exists
PREPROCESSED_PATH.mkdir(parents=True, exist_ok=True)


def get_aalto_files():
    """Return list of Aalto keystroke files."""
    if not AALTO_RAW_PATH.exists():
        raise FileNotFoundError(f"Aalto dataset not found at {AALTO_RAW_PATH}")
    return sorted(AALTO_RAW_PATH.glob("*_keystrokes.txt"))


def get_user_id_from_filename(filename):
    """Extract user ID from filename like '100001_keystrokes.txt'."""
    return filename.stem.replace("_keystrokes", "")
