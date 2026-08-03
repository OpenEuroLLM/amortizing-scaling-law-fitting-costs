"""Fetch the raw source files from their origin repos.

GIT_SOURCES are pulled via a sparse, depth-1 clone of just the one file's
directory, using whatever git credentials are already configured locally
(e.g. Windows Credential Manager) -- the same way a normal `git clone` of
a private repo would work. This avoids hardcoding tokens and works
uniformly whether the source repo is public or private.

URL_SOURCES are single files at a stable public URL, fetched with a plain
HTTP download -- no need for git's sparse-clone machinery.
"""
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"

GIT_SOURCES = {
    "oellm_english": {
        "repo": "https://github.com/OpenEuroLLM/dense_english_scaling_results.git",
        "path": "data/preprocessed/all_points_clean.csv",
    },
    "oellm_multilingual": {
        "repo": "https://github.com/OpenEuroLLM/dense_multilingual_models_scaling_results.git",
        "path": "scaling_laws/data/preprocessed/decay_points_clean.csv",
    },
    "steplaw": {
        "repo": "https://github.com/step-law/steplaw.git",
        "path": "data/dense_lr_bs_loss.csv",
    },
}

URL_SOURCES = {
    "porian": {
        "url": "https://raw.githubusercontent.com/formll/resolving-scaling-law-discrepancies/main/data/experiment_results.pickle.xz",
        "filename": "experiment_results.pickle.xz",
    },
    "gemstones": {
        "url": "https://raw.githubusercontent.com/mcleish7/gemstone-scaling-laws/main/wandb_dfs/wandb_df_for_fitting_hot.jsonl",
        "filename": "wandb_df_for_fitting_hot.jsonl",
    },
}


def fetch_git(name: str, repo: str, path: str) -> Path:
    dest = RAW_DIR / name / Path(path).name
    dest.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", repo, tmp],
            check=True,
        )
        subprocess.run(
            ["git", "-C", tmp, "sparse-checkout", "set", str(Path(path).parent)],
            check=True,
        )
        shutil.copyfile(Path(tmp) / path, dest)

    return dest


def fetch_url(name: str, url: str, filename: str) -> Path:
    dest = RAW_DIR / name / filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    return dest


def main():
    for name, cfg in GIT_SOURCES.items():
        dest = fetch_git(name, cfg["repo"], cfg["path"])
        print(f"{name}: fetched -> {dest}")
    for name, cfg in URL_SOURCES.items():
        dest = fetch_url(name, cfg["url"], cfg["filename"])
        print(f"{name}: fetched -> {dest}")


if __name__ == "__main__":
    main()
