"""
Inspect Aalto dataset structure and provide summary statistics.

Usage: python data/inspect.py
"""

from pathlib import Path
from config import AALTO_RAW_PATH, get_aalto_files


def inspect_dataset():
    """Print dataset structure and sample file contents."""
    files = get_aalto_files()
    print(f"✓ Found {len(files)} keystroke files")
    print(f"  Location: {AALTO_RAW_PATH}")
    print()

    # Sample a file
    sample_file = files[0]
    print(f"Sample file: {sample_file.name}")
    print(f"File size: {sample_file.stat().st_size:,} bytes")
    print()

    # Read and inspect format
    print("First 3 lines of sample file:")
    with open(sample_file) as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            print(f"  {line.rstrip()}")
    print()

    # Count unique subjects (user IDs)
    user_ids = set()
    for file in files:
        user_id = file.stem.replace("_keystrokes", "")
        user_ids.add(user_id)

    print(f"Unique subjects (user IDs): {len(user_ids)}")
    print(f"Average files per subject: {len(files) / len(user_ids):.1f}")


if __name__ == "__main__":
    inspect_dataset()
