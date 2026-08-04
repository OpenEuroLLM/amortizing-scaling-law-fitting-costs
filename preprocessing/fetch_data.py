"""Fetch raw source files from their origin repositories.

GIT_SOURCES are pulled via a sparse, depth-1 clone of a single file's
directory. URL_SOURCES are downloaded directly over HTTP.
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
    # Li et al., "Predictable Scale: Part I, Step Law -- Optimal Hyperparameter
    # Scaling Law in Large Language Model Pretraining", arXiv:2503.04715.
    "steplaw": {
        "repo": "https://github.com/step-law/steplaw.git",
        "path": "data/dense_lr_bs_loss.csv",
    },
}

URL_SOURCES = {
    # Porian et al., "Resolving Discrepancies in Compute-Optimal Scaling of
    # Language Models", arXiv:2406.19146.
    "porian": {
        "url": "https://raw.githubusercontent.com/formll/resolving-scaling-law-discrepancies/main/data/experiment_results.pickle.xz",
        "filename": "experiment_results.pickle.xz",
    },
}

# Sources fetched as multiple files sharing a base URL, downloaded into one
# raw dir (preprocess.py then adapts each file separately or concatenates
# them, depending on the source -- see ADAPTERS there).
SPLIT_URL_SOURCES = {
    # HuggingFace-style train/test parquet splits, from the benchmark suite of
    # "Spend Less, Fit Better: Budget-Efficient Scaling Law Fitting via Active
    # Experiment Selection" (arXiv:2604.22753). Only the sub-datasets whose
    # columns map onto N/D/lr/gbsz/loss (or a subset of it) are included here
    # -- see NOTES.md for the other sub-datasets and why they don't fit.
    "activesl_lr_bsz": {
        "base_url": "https://raw.githubusercontent.com/PlanarG/active-sl/main/benchmark/dataset/lr_bsz_scaling_law",
        "filenames": ["train-00000-of-00001.parquet", "test-00000-of-00001.parquet"],
    },
    "activesl_farseer": {
        "base_url": "https://raw.githubusercontent.com/PlanarG/active-sl/main/benchmark/dataset/farseer_scaling_law",
        "filenames": ["train-00000-of-00001.parquet", "test-00000-of-00001.parquet"],
    },
    "activesl_data_constrained": {
        "base_url": "https://raw.githubusercontent.com/PlanarG/active-sl/main/benchmark/dataset/data_constrained_scaling_law",
        "filenames": ["train-00000-of-00001.parquet", "test-00000-of-00001.parquet"],
    },
    # McLeish et al., "Gemstones: A Model Suite for Multi-Faceted Scaling
    # Laws", arXiv:2502.06857. Each file is a distinct dataset variant (token
    # range / checkpoint density / LR schedule), adapted separately in
    # preprocess.py -- see data/raw/gemstones/README.md for what each one is.
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


def fetch_split(name: str, base_url: str, filenames: list) -> Path:
    dest_dir = RAW_DIR / name
    dest_dir.mkdir(parents=True, exist_ok=True)
    for filename in filenames:
        urllib.request.urlretrieve(f"{base_url}/{filename}", dest_dir / filename)
    return dest_dir


def main():
    for name, cfg in GIT_SOURCES.items():
        dest = fetch_git(name, cfg["repo"], cfg["path"])
        print(f"{name}: fetched -> {dest}")
    for name, cfg in URL_SOURCES.items():
        dest = fetch_url(name, cfg["url"], cfg["filename"])
        print(f"{name}: fetched -> {dest}")
    for name, cfg in SPLIT_URL_SOURCES.items():
        dest = fetch_split(name, cfg["base_url"], cfg["filenames"])
        print(f"{name}: fetched -> {dest}")


if __name__ == "__main__":
    main()
