"""Selective per-ablation data fetch for ``ablation_analysis.ipynb``.

Each ablation's runs are packaged as one zip on Google Drive (see ``ablations_manifest.json``).
``fetch("baselines")`` resolves *one* ablation to a local folder — pulling only that zip — so a
collaborator working on a single section never downloads the whole study. Resolution order:

1. **local checkout** (default) — a sibling ``experiments-ablations/<dir>`` with a ``runs/`` folder,
   so the author (who already has the data) runs the notebook with no download. Point elsewhere with
   ``$ABLATIONS_LOCAL`` or ``local_root=``.
2. **cache** — a previously downloaded+unzipped copy under ``analysis/_ablation_cache/``.
3. **Drive** — download the ablation's zip by ``drive_id`` (needs ``gdown``; fill the id in the
   manifest after uploading), unzip into the cache.

Usage in the notebook:

    from ablation_fetch import fetch, runs_dir, available
    rd = runs_dir("baselines")          # -> Path to .../ablation_baselines/runs
"""
from __future__ import annotations
import json, os, zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "ablations_manifest.json"
CACHE = HERE / "_ablation_cache"
REPO = HERE.parents[2]                       # experiments/workshop_result/analysis -> repo root is 3 up
LOCAL_DEFAULT = REPO.parent / "experiments-ablations"   # sibling checkout of the raw ablation folders


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text())


def available() -> list[str]:
    return list(load_manifest()["ablations"])


def _entry(name: str) -> dict:
    m = load_manifest()
    if name not in m["ablations"]:
        raise KeyError(f"unknown ablation {name!r}; choices: {available()}")
    return m["ablations"][name]


def _local_dir(entry: dict, local_root=None) -> Path | None:
    if local_root is None:
        local_root = os.environ.get("ABLATIONS_LOCAL", LOCAL_DEFAULT)
    p = Path(local_root) / entry["dir"]
    return p if (p / "runs").is_dir() else None


def fetch(name: str, *, force: bool = False, prefer_local: bool = True, local_root=None) -> Path:
    """Return a local folder for ablation ``name`` (contains ``runs/``), fetching if needed."""
    entry = _entry(name)

    if prefer_local and not force:
        loc = _local_dir(entry, local_root)
        if loc is not None:
            return loc

    dest = CACHE / entry["dir"]
    if dest.is_dir() and (dest / "runs").is_dir() and not force:
        return dest

    drive_id = entry.get("drive_id")
    if not drive_id:
        raise RuntimeError(
            f"no drive_id for {name!r} in {MANIFEST.name} and no local checkout found. "
            f"Upload {entry['zip']} to Drive and record its file id, or set "
            f"ABLATIONS_LOCAL to a folder containing {entry['dir']}/runs.")
    try:
        import gdown
    except ImportError as e:
        raise RuntimeError("gdown is required to download from Drive: pip install gdown") from e

    CACHE.mkdir(parents=True, exist_ok=True)
    zpath = CACHE / entry["zip"]
    gdown.download(id=drive_id, output=str(zpath), quiet=False)
    with zipfile.ZipFile(zpath) as zf:
        zf.extractall(CACHE)
    zpath.unlink()
    if not (dest / "runs").is_dir():
        raise RuntimeError(f"{entry['zip']} did not unpack to {dest}/runs as expected")
    return dest


def runs_dir(name: str, **kw) -> Path:
    return fetch(name, **kw) / "runs"
