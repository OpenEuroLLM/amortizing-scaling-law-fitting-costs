"""Fetch raw source files from their origin repositories.

GIT_SOURCES are pulled via a sparse, depth-1 clone of a single file's
directory. URL_SOURCES are downloaded directly over HTTP.
"""
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

GIT_SOURCES = {
    # TODO: add a link to the tech report once it's published.
    "oellm_english": {
        "repo": "https://github.com/OpenEuroLLM/dense_english_scaling_results.git",
        "paths": ["data/preprocessed/all_points_clean.csv"],
    },
    # TODO: add a link to the tech report once it's published.
    "oellm_multilingual": {
        "repo": "https://github.com/OpenEuroLLM/dense_multilingual_models_scaling_results.git",
        "paths": ["scaling_laws/data/preprocessed/decay_points_clean.csv"],
    },
    # Li et al., "Predictable Scale: Part I, Step Law -- Optimal Hyperparameter
    # Scaling Law in Large Language Model Pretraining", arXiv:2503.04715.
    "steplaw": {
        "repo": "https://github.com/step-law/steplaw.git",
        "paths": ["data/dense_lr_bs_loss.csv", "data/moe_lr_bs_loss.csv"],
    },
}

URL_SOURCES = {
    # Porian et al., "Resolving Discrepancies in Compute-Optimal Scaling of
    # Language Models", arXiv:2406.19146.
    "porian": {
        "url": "https://raw.githubusercontent.com/formll/resolving-scaling-law-discrepancies/main/data/experiment_results.pickle.xz",
        "filename": "experiment_results.pickle.xz",
    },
    # Mallik et al., "When is Warmstarting Effective for Scaling Language
    # Models?", arXiv:2605.13405. Drive link the authors share for the raw
    # data in a sibling project's: EffingScaling/README.md; "Warmstarting Data".
    "warmstart": {
        "url": "https://drive.google.com/uc?export=download&id=1vr5Jq1TpTpnkb5CX5mcu2wWQQzbf-bmd",
        "filename": "warmstart_runs_flattened.parquet",
    },
}

SPLIT_URL_SOURCES = {
    # McLeish et al., "Gemstones: A Model Suite for Multi-Faceted Scaling
    # Laws", arXiv:2502.06857. Each file corresponds to a distinct scaling
    # experiment, adapted separately in preprocess.py -- see
    # data/gemstones/README.md for what each one is.
    "gemstones": {
        "base_url": "https://raw.githubusercontent.com/mcleish7/gemstone-scaling-laws/main/wandb_dfs",
        "filenames": [
            "wandb_df_for_fitting_hot.jsonl",
            "wandb_df_for_fitting_hot_477.jsonl",
            "wandb_df_for_fitting_hot_100b+.jsonl",
            "wandb_df_for_fitting_hot_100b+_477.jsonl",
            "wandb_df_for_fitting_hot_120b+_only.jsonl",
            "wandb_df_for_fitting_hot_120b+_only_477.jsonl",
            "wandb_df_for_fitting_hot_100b+_512x.jsonl",
            "wandb_df_for_fitting_cool_begin.jsonl",
            "wandb_df_for_fitting_cool_end.jsonl",
            "wandb_df_for_fitting_lr_ablation_hot.jsonl",
            "wandb_df_for_fitting_lr_ablation_hot_477.jsonl",
            "wandb_df_for_fitting_hot_sampled_like_chinchilla.jsonl",
            "wandb_df_for_fitting_lr_ablation_hot_sampled_like_chinchilla.jsonl",
            "wandb_df_for_fitting_slim_chinchilla.jsonl",
            "wandb_df_for_fitting_lr_ablation_slim_chinchilla.jsonl",
        ],
    },
}


def fetch_git(name: str, repo: str, paths: list) -> Path:
    dest_dir = DATA_DIR / name / "raw"
    dest_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", repo, tmp],
            check=True,
        )
        subprocess.run(
            ["git", "-C", tmp, "sparse-checkout", "set"]
            + [str(Path(path).parent) for path in paths],
            check=True,
        )
        for path in paths:
            shutil.copyfile(Path(tmp) / path, dest_dir / Path(path).name)

    return dest_dir


def fetch_url(name: str, url: str, filename: str) -> Path:
    dest = DATA_DIR / name / "raw" / filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    return dest


def fetch_split(name: str, base_url: str, filenames: list) -> Path:
    dest_dir = DATA_DIR / name / "raw"
    dest_dir.mkdir(parents=True, exist_ok=True)
    for filename in filenames:
        urllib.request.urlretrieve(f"{base_url}/{filename}", dest_dir / filename)
    return dest_dir


def main():
    for name, cfg in GIT_SOURCES.items():
        dest = fetch_git(name, cfg["repo"], cfg["paths"])
        print(f"{name}: fetched -> {dest}")
    for name, cfg in URL_SOURCES.items():
        dest = fetch_url(name, cfg["url"], cfg["filename"])
        print(f"{name}: fetched -> {dest}")
    for name, cfg in SPLIT_URL_SOURCES.items():
        dest = fetch_split(name, cfg["base_url"], cfg["filenames"])
        print(f"{name}: fetched -> {dest}")


if __name__ == "__main__":
    main()
