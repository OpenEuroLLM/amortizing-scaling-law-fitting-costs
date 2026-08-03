"""Transform each raw source CSV into the common N, D, lr, gbsz, loss schema
that iterative_fits_L0_form.py and model_based_fits.py expect as --input.
"""
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
OUT_DIR = Path(__file__).parent.parent / "data" / "preprocessed"

COLUMNS = ["N", "D", "lr", "gbsz", "loss"]

# (depth, width) -> parameter count in millions, from Table 2, Appendix B of
# Porian et al. (arXiv:2406.19146). Four counting conventions, per the paper's
# Section 2.1 / Table 2 footnote (embedding and unembedding/head are tied):
#   N_Porian: the paper's own default "N" -- parameters in all linear layers,
#             excluding the embedding layer but including the head. Since the
#             head reuses the tied embedding matrix, this ends up almost equal
#             to N_exact (the exclusion/inclusion cancel out).
#   N_exact:  the exact total trainable parameter count of the model (every
#             parameter, including the tied embedding/head matrix once).
#   N_eff:    "effective" parameters -- N_Porian adjusted upward to also
#             account for attention-operation FLOPs, which raw parameter
#             counts don't capture; bigger relative effect on smaller models.
#   N_Kaplan: Kaplan et al.'s original convention -- excludes both the
#             embedding layer and the head, so it drops the large tied
#             embedding/unembedding matrix entirely. Massively undercounts
#             small models (up to ~90% smaller here) since that matrix
#             dominates their parameter count.
PORIAN_PARAM_LOOKUP_M = {
    (3, 96): {"N_Porian": 5.173, "N_exact": 5.176, "N_eff": 5.763, "N_Kaplan": 0.331},
    (4, 128): {"N_Porian": 7.504, "N_exact": 7.508, "N_eff": 8.552, "N_Kaplan": 1.049},
    (5, 160): {"N_Porian": 9.810, "N_exact": 9.817, "N_eff": 11.45, "N_Kaplan": 1.741},
    (6, 224): {"N_Porian": 15.60, "N_exact": 15.61, "N_eff": 18.35, "N_Kaplan": 4.301},
    (8, 288): {"N_Porian": 22.49, "N_exact": 22.51, "N_eff": 27.21, "N_Kaplan": 7.963},
    (9, 320): {"N_Porian": 28.67, "N_exact": 28.70, "N_eff": 34.57, "N_Kaplan": 12.53},
    (10, 384): {"N_Porian": 37.06, "N_exact": 37.09, "N_eff": 44.92, "N_Kaplan": 17.69},
    (12, 480): {"N_Porian": 57.38, "N_exact": 57.43, "N_eff": 69.18, "N_Kaplan": 33.18},
    (14, 576): {"N_Porian": 84.79, "N_exact": 84.85, "N_eff": 101.3, "N_Kaplan": 55.74},
    (15, 640): {"N_Porian": 108.5, "N_exact": 108.5, "N_eff": 128.1, "N_Kaplan": 76.19},
    (18, 704): {"N_Porian": 149.0, "N_exact": 149.1, "N_eff": 175.0, "N_Kaplan": 113.5},
    (21, 832): {"N_Porian": 220.9, "N_exact": 221.0, "N_eff": 256.7, "N_Kaplan": 178.9},
    (23, 1024): {"N_Porian": 347.1, "N_exact": 347.3, "N_eff": 395.3, "N_Kaplan": 295.4},
    (26, 1120): {"N_Porian": 455.3, "N_exact": 455.5, "N_eff": 514.9, "N_Kaplan": 398.8},
    (26, 1312): {"N_Porian": 612.0, "N_exact": 612.2, "N_eff": 681.8, "N_Kaplan": 545.8},
    (30, 1504): {"N_Porian": 901.7, "N_exact": 902.1, "N_eff": 994.1, "N_Kaplan": 825.9},
}
PORIAN_N_VARIANTS = ["N_Porian", "N_exact", "N_eff", "N_Kaplan"]


def adapt_oellm_english(df: pd.DataFrame) -> pd.DataFrame:
    return df[COLUMNS]


def adapt_oellm_multilingual(df: pd.DataFrame) -> pd.DataFrame:
    # N is ambiguous in this source (N_ne_*, N_total_*, N_Porian_*); using the
    # Porian et al. convention -- see NOTES.md for the open question on
    # verifying this choice (and D/lr/gbsz) against each source's intent.
    df = df.rename(columns={"N_Porian_exact": "N"})
    return df[COLUMNS]


def adapt_steplaw(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={"bs": "gbsz"})
    return df[COLUMNS]


def adapt_porian(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={"bs": "gbsz"})
    df["loss"] = df["val/loss"].apply(lambda s: s.iloc[-1])
    df["D"] = df["seq_len"] * df["gbsz"] * df["max_step"]
    for n_col in PORIAN_N_VARIANTS:
        df[n_col] = [
            PORIAN_PARAM_LOOKUP_M[(depth, width)][n_col] * 1e6
            for depth, width in zip(df["depth"], df["width"])
        ]
    # N_Porian is the default "N" for now, so this is directly usable with
    # --input; N_exact/N_eff/N_Kaplan stay available as their own columns
    # until the fitting scripts support choosing a variant (NOTES.md to-do).
    df["N"] = df["N_Porian"]
    return df[COLUMNS + ["N_Porian", "N_exact", "N_eff", "N_Kaplan"]]


def adapt_gemstones(df: pd.DataFrame) -> pd.DataFrame:
    # No lr/gbsz anywhere in this source (main sweep uses a muP-derived LR
    # rule that isn't logged per-run); usable for envelope fits,
    # not for the surrogate scripts that need all 4 features.
    df = df.rename(columns={"params_active_precise": "N", "tokens": "D", "final_loss": "loss"})
    df["lr"] = None
    df["gbsz"] = None
    return df[COLUMNS]


def adapt_activesl_lr_bsz(df: pd.DataFrame) -> pd.DataFrame:
    # From the "lr_bsz_scaling_law" config of the active-sl benchmark (arXiv:2604.22753).
    # "group" is always "all_data" in this config -- not informative, dropped.
    df = df.rename(columns={
        "bsz": "gbsz",
        "data_size": "D",
        "non_embedding_param_size": "N",
        "lm_loss": "loss",
    })
    return df[COLUMNS]


def adapt_activesl_farseer(df: pd.DataFrame) -> pd.DataFrame:
    # From the "farseer_scaling_law" config of the active-sl benchmark. Already
    # N/D/loss; no lr/gbsz in this config.
    df["lr"] = None
    df["gbsz"] = None
    return df[COLUMNS]


def adapt_activesl_data_constrained(df: pd.DataFrame) -> pd.DataFrame:
    # From the "data_constrained_scaling_law" config of the active-sl benchmark.
    # No lr/gbsz. "unique_tokens" (distinct, non-repeated tokens; D can exceed
    # it when a run repeats data for multiple epochs) has no slot in the
    # common schema, so it's kept as an extra column.
    df = df.rename(columns={"params": "N", "tokens": "D"})
    df["lr"] = None
    df["gbsz"] = None
    return df[COLUMNS + ["unique_tokens"]]


def adapt_warmstart(df: pd.DataFrame) -> pd.DataFrame:
    # Per-step logs (one row per eval, ~2.4M rows); collapsed to one row per
    # run -- identified by its `post_hoc_df` output path -- using the last
    # checkpoint that has a logged Validation Loss (a couple of runs are
    # missing their final eval, so this isn't always literally the last step).
    # N is `target_N`, the post-growth size actually being evaluated;
    # `base_N` (the pre-growth size the run warmstarted from), `shrink`,
    # `warmstart_type`, and `method` are kept as extra columns since they
    # identify the warmstarting technique, not something the common schema
    # has a slot for.
    df = df.dropna(subset=["Validation Loss"]).sort_values("tokens")
    df = df.groupby("post_hoc_df", as_index=False).tail(1)
    df = df.rename(columns={
        "target_N": "N",
        "tokens": "D",
        "max_lr": "lr",
        "Validation Loss": "loss",
    })
    df["gbsz"] = df["micro_batch_size"] * df["accumulation_iters"] * df["devices"]
    return df[COLUMNS + ["base_N", "shrink", "warmstart_type", "method"]]


# name -> (adapter, raw filename(s), file format). filename(s) is a list for
# sources fetched as multiple files (e.g. HuggingFace train/test splits) that
# get concatenated before adapting.
ADAPTERS = {
    "oellm_english": (adapt_oellm_english, "all_points_clean.csv", "csv"),
    "oellm_multilingual": (adapt_oellm_multilingual, "decay_points_clean.csv", "csv"),
    "steplaw": (adapt_steplaw, "dense_lr_bs_loss.csv", "csv"),
    "porian": (adapt_porian, "experiment_results.pickle.xz", "pickle"),
    "gemstones": (adapt_gemstones, "wandb_df_for_fitting_hot.jsonl", "jsonl"),
    "activesl_lr_bsz": (
        adapt_activesl_lr_bsz,
        ["train-00000-of-00001.parquet", "test-00000-of-00001.parquet"],
        "parquet",
    ),
    "activesl_farseer": (
        adapt_activesl_farseer,
        ["train-00000-of-00001.parquet", "test-00000-of-00001.parquet"],
        "parquet",
    ),
    "activesl_data_constrained": (
        adapt_activesl_data_constrained,
        ["train-00000-of-00001.parquet", "test-00000-of-00001.parquet"],
        "parquet",
    ),
    # No public source -- see NOTES.md; raw file must be placed manually at
    # data/raw/warmstart/warmstart_runs_flattened.parquet.
    "warmstart": (adapt_warmstart, "warmstart_runs_flattened.parquet", "parquet"),
}

READERS = {
    "csv": pd.read_csv,
    "pickle": pd.read_pickle,
    "jsonl": lambda path: pd.read_json(path, lines=True),
    "parquet": pd.read_parquet,
}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, (adapt, filenames, filetype) in ADAPTERS.items():
        if isinstance(filenames, list):
            df = pd.concat(
                [READERS[filetype](RAW_DIR / name / f) for f in filenames],
                ignore_index=True,
            )
        else:
            df = READERS[filetype](RAW_DIR / name / filenames)
        out = adapt(df)
        # some sources store N/D as int64 (e.g. steplaw); 6*N*D can overflow int64
        # at large scales and silently wrap negative, so enforce float64 here.
        float_cols = [c for c in out.columns if c == "D" or c.startswith("N")]
        out = out.astype({c: "float64" for c in float_cols})
        out_path = OUT_DIR / f"{name}.csv"
        out.to_csv(out_path, index=False)
        print(f"{name}: {len(out)} rows -> {out_path}")


if __name__ == "__main__":
    main()
