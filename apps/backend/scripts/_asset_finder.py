"""
Asset discovery utility for transformer model and test data.

Searches common locations in priority order:
1. ./artifacts/
2. Repo root
3. data/ directory
4. Full repo search (excluding node_modules, .git, etc.)
"""
from pathlib import Path
from typing import Optional


def find_asset(filename: str, search_dirs: list[Path] | None = None) -> Path:
    """
    Search for asset file in common locations.

    Args:
        filename: Name of the file to find
        search_dirs: Optional list of directories to search first

    Returns:
        Path to the found file

    Raises:
        FileNotFoundError: If file not found, with list of searched paths
    """
    # scripts/_asset_finder.py -> apps/backend/scripts -> apps/backend -> apps -> repo_root
    repo_root = Path(__file__).resolve().parents[3]

    default_dirs = [
        repo_root / "artifacts",
        repo_root,
        repo_root / "data",
        repo_root / "transformer",
    ]

    search_dirs = search_dirs or default_dirs
    searched: list[str] = []

    # Check priority directories first
    for d in search_dirs:
        if d.exists():
            candidate = d / filename
            if candidate.exists():
                return candidate
            searched.append(str(d))

    # Fallback: recursive search (slow but thorough)
    excluded = {"node_modules", ".git", "__pycache__", ".venv", "venv", ".tox"}
    for candidate in repo_root.rglob(filename):
        if not any(p in candidate.parts for p in excluded):
            return candidate

    raise FileNotFoundError(
        f"Asset '{filename}' not found.\n"
        f"Searched directories: {searched}\n"
        f"Also performed recursive search from: {repo_root}"
    )


def find_transformer_zip() -> Path:
    """Find the transformer model zip archive."""
    return find_asset("transformer-20260121T131215Z-3-001.zip")


def find_y_test_npy() -> Path:
    """Find the y_test.npy fixture data file."""
    return find_asset("y_test.npy")


if __name__ == "__main__":
    # Quick test
    try:
        print(f"Transformer zip: {find_transformer_zip()}")
    except FileNotFoundError as e:
        print(f"Not found: {e}")

    try:
        print(f"y_test.npy: {find_y_test_npy()}")
    except FileNotFoundError as e:
        print(f"Not found: {e}")
