import os


def portable_path(path: str, root: str) -> str:
    """Store project-local paths relatively so model configs can move."""
    abs_path = os.path.abspath(path)
    abs_root = os.path.abspath(root)
    try:
        common = os.path.commonpath([abs_path, abs_root])
    except ValueError:
        return abs_path
    if common != abs_root:
        return abs_path
    return os.path.relpath(abs_path, abs_root)


def resolve_config_path(path: str | None, root: str, default: str) -> str:
    if not path:
        return default
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(root, path))
