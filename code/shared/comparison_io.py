"""Output isolation shared by comparison calculation and figure entry points."""
from pathlib import Path


def prepare_output_directory(path):
    """Create an empty destination, refusing to mix or overwrite previous runs."""
    path = Path(path).resolve()
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"Output must be a new or empty directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path
