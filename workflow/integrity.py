"""Deterministic file and directory integrity records used across stages."""
import hashlib
import json
from pathlib import Path


def sha256_file(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def artifact_digest(path):
    """Return a deterministic digest for a file or a directory tree."""
    path = Path(path).resolve()
    if path.is_file():
        return {"kind": "file", "size": path.stat().st_size, "sha256": sha256_file(path)}
    if not path.is_dir():
        raise FileNotFoundError(f"artifact does not exist: {path}")
    files = []
    for child in sorted(p for p in path.rglob("*") if p.is_file()):
        files.append({"path": child.relative_to(path).as_posix(), "size": child.stat().st_size,
                      "sha256": sha256_file(child)})
    payload = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {"kind": "directory", "size": sum(item["size"] for item in files),
            "sha256": hashlib.sha256(payload).hexdigest(), "files": files}


def verify_artifact(path, expected):
    actual = artifact_digest(path)
    for key in ("kind", "size", "sha256"):
        if actual.get(key) != expected.get(key):
            raise RuntimeError(f"artifact integrity check failed: {Path(path).resolve()}")
    return actual
