"""Transform each raw source file into the common N, D, lr, gbsz, loss schema that
iterative_fits_L0_form.py and model_based_fits.py expect as --input.
"""
import argparse
import math
from functools import partial
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
OUT_DIR = Path(__file__).parent.parent / "data" / "preprocessed"

COLUMNS = ["N", "D", "lr", "gbsz", "loss"]

# (depth, width) -> parameter count in millions, from Table 2, Appendix B of
# Porian et al. (arXiv:2406.19146), in the paper's own N_Porian/N_exact/N_eff/
# N_Kaplan conventions -- see README.md's "N-variant vocabulary" section.
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


# Table 1 ("Model architectures"), Sec. 1.1 of Ajroldi, "Scaling of
# OpenEuroLLM Dense English Models" (see data/raw/oellm_english/README.md),
# keyed by exact N (non-embedding parameter count -- matches Table 1's "Total
# parameters" row exactly, and is independently reconstructed by
# vocab_size(50,000) * width == the embedding-matrix size subtracted out in
# scripts/preprocess.ipynb's model_sizes dict, for all 5 sizes). The raw
# file's held-out 1.7B rows aren't in Table 1, so are left without an
# architecture (None).
OELLM_ENGLISH_ARCH = {
    47_557_632: {"width": 384, "depth": 12, "num_heads": 6, "ffn": 1536},
    124_455_168: {"width": 576, "depth": 18, "num_heads": 9, "ffn": 2304},
    301_880_320: {"width": 896, "depth": 20, "num_heads": 14, "ffn": 3584},
    588_544_000: {"width": 1280, "depth": 20, "num_heads": 20, "ffn": 5120},
    983_138_304: {"width": 1536, "depth": 24, "num_heads": 24, "ffn": 6144},
}


def adapt_oellm_english(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    arch = df["N"].map(OELLM_ENGLISH_ARCH)
    for key in ("width", "depth", "num_heads", "ffn"):
        df[key] = arch.apply(lambda a, key=key: a[key] if isinstance(a, dict) else None)
    df["tied_embeddings"] = True  # Sec. 1.1: "tied input/output embeddings"
    df["seq_len"] = 4096  # scripts/preprocess.ipynb SEQ_LENGTH; reconstructs train_tokens exactly
    df["N_definition"] = "N_ne"
    df["warmup_iters"] = 2000  # fixed across all runs (Sec. 1.1 / scripts/clean.ipynb)
    df["lr_schedule"] = "linear_decay"  # WSD decay phase is linear (confirmed)
    # decay covers the final 20% of each run's own steps (Sec. 3, "WSD
    # Analysis"); `iteration` is the run's final step, matching
    # scripts/preprocess.ipynb's own branching-checkpoint formula
    # (floor(iteration * 0.8)).
    decay_start = (df["iteration"] * 0.8).apply(math.floor)
    df["decay_iters"] = df["iteration"] - decay_start
    df["min_lr"] = 1e-5  # Sec. 3: decay always anneals to a fixed floor lr of 1e-5
    return df[
        COLUMNS
        + ["width", "depth", "tied_embeddings", "seq_len", "N_definition"]
        + ["lr_schedule", "warmup_iters", "decay_iters", "min_lr"]
        + ["num_heads", "ffn", "beta2", "seed"]
    ]


def adapt_oellm_multilingual(df: pd.DataFrame) -> pd.DataFrame:
    # N defaults to N_Porian_exact, matching this source's own repo-root
    # README.md ("N_Porian" convention) and its utils.py (N_Porian_exact ==
    # N_total_exact here, since all runs use tied embeddings); N_ne/N_total
    # are kept as their own columns too, using the project's N-variant names
    # (README.md's "N-variant vocabulary") -- this source's own README
    # confirms N_ne/N_total/N_Porian match those definitions directly.
    df = df.rename(columns={
        "N_Porian_exact": "N",
        "N_ne_exact": "N_ne",
        "N_total_exact": "N_total",
    })
    df["tied_embeddings"] = True  # repo README: untie_embeddings_and_output_weights: False
    df["seq_len"] = 4096  # preprocess_decay.py SEQ_LEN / hp_selection/utils.py compute_checkpoint_iter
    df["N_definition"] = "N_Porian"
    df["lr_schedule"] = "linear_decay"  # WSD decay phase is linear (confirmed)
    df["warmup_iters"] = 2000  # fixed across all runs (confirmed)
    # decay covers the final 20% of each run's own steps (hp_selection/utils.py
    # compute_checkpoint_iter's decay_fraction=0.2, "constant across the whole
    # multilingual_scaling model family"); `iteration` is the run's final step
    # (decay_points_clean.csv already filters to iteration == expected_iters).
    decay_start = (df["iteration"] * 0.8).apply(math.floor)
    df["decay_iters"] = df["iteration"] - decay_start
    df["min_lr"] = 0  # decay floor (confirmed)
    return df[
        COLUMNS
        + ["N_ne", "N_total", "tied_embeddings", "seq_len", "N_definition"]
        + ["lr_schedule", "warmup_iters", "decay_iters", "min_lr"]
    ]


def adapt_steplaw_dense(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={
        "bs": "gbsz",
        "h": "width",
        "numl": "depth",
        "ffnh": "ffn",
        "numh": "num_heads",
    })
    df["seq_len"] = 2048
    df["warmup_iters"] = 2000
    df["lr_schedule"] = "cosine decay"
    df["N_definition"] = "N_ne"
    return df[
        COLUMNS
        + ["width", "depth", "ffn", "num_heads", "seq_len", "warmup_iters", "lr_schedule", "N_definition"]
    ]


def adapt_steplaw_moe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={
        "bs": "gbsz",
        "h": "width",
        "numl": "depth",
        "ffnh": "ffn",
        "numh": "num_heads",
        "nume": "num_experts",
        "moeh": "expert_width",
        "topk": "top_k",
        "Na": "N_active",
    })
    df["warmup_iters"] = 2000
    df["lr_schedule"] = "cosine decay"
    df["N_definition"] = "N_ne"
    return df[
        COLUMNS
        + ["width", "depth", "ffn", "num_heads", "seq_len", "warmup_iters", "lr_schedule"]
        + ["num_experts", "expert_width", "top_k", "N_active", "N_definition"]
    ]


def adapt_porian(df: pd.DataFrame, hparams: str, dataset: str, decay: str | None = None) -> pd.DataFrame:
    # The raw file bundles runs from several experiments; each output dataset
    # filters to one (hparams, dataset, decay) combination -- see
    # data/raw/porian/README.md.
    mask = (df["hparams"] == hparams) & (df["dataset"] == dataset)
    if decay is not None:
        mask &= df["decay"] == decay
    df = df[mask]
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
    # until the fitting scripts support choosing a variant.
    df["N"] = df["N_Porian"]
    # warmup_iters: warmup_tokens (raw) converted to steps at this run's own
    # seq_len/gbsz.
    df["warmup_iters"] = (df["warmup_tokens"] / (df["seq_len"] * df["gbsz"])).apply(math.ceil)
    # decay_iters/min_lr: derived per raw "decay" value, verified against
    # the raw per-step "train/lr" curves. "const" (Sec. 3.5) holds the LR
    # flat after warmup -- no decay, so decay_iters/min_lr don't apply
    # (None, as for gemstones' non-decaying rows). "chinchilla" (Sec. 3.4)
    # cosine-decays from the end of warmup to exactly 1% of peak lr at the
    # run's own max_step -- confirmed across the whole chinchilla set:
    # floor/peak converges to ~0.01 right at max_step regardless of
    # max_step's value, so decay_iters = max_step - warmup_iters and
    # min_lr = 0.01 * lr. "kaplan" (Sec. 3.1) cosine-decays to exactly 0 at
    # a fixed token target (2.5e5 * 2**19 ~= 131B tokens) shared by every
    # kaplan run regardless of model size -- confirmed by reproducing each
    # run's logged floor value from a cosine formula using that shared
    # target (matches to 6 decimal places across all 64 rows).
    kaplan_decay_end_tokens = 2.5e5 * 2**19
    df["decay_iters"] = float("nan")
    df["min_lr"] = float("nan")
    is_chinchilla = df["decay"] == "chinchilla"
    if is_chinchilla.any():
        df.loc[is_chinchilla, "decay_iters"] = (
            df.loc[is_chinchilla, "max_step"] - df.loc[is_chinchilla, "warmup_iters"]
        )
        df.loc[is_chinchilla, "min_lr"] = 0.01 * df.loc[is_chinchilla, "lr"]
    is_kaplan = df["decay"] == "kaplan"
    if is_kaplan.any():
        df.loc[is_kaplan, "decay_iters"] = (
            (kaplan_decay_end_tokens - df.loc[is_kaplan, "warmup_tokens"])
            / (df.loc[is_kaplan, "seq_len"] * df.loc[is_kaplan, "gbsz"])
        ).apply(math.ceil)
        df.loc[is_kaplan, "min_lr"] = 0.0
    df["lr_schedule"] = df["decay"].map({"const": "stable", "chinchilla": "cosine_decay", "kaplan": "cosine_decay"})
    # tied_embeddings: not a raw column: confirmed by reconstructing N_exact as
    # N_Kaplan (non-embedding params) + one vocab*width embedding matrix, which
    # matches the lookup table to within ~0.1% across all 16 architectures --
    # two matrices (untied) would overshoot N_exact by 8-94%.
    df["tied_embeddings"] = True
    return df[
        COLUMNS
        + ["N_Porian", "N_exact", "N_eff", "N_Kaplan"]
        + ["width", "depth", "seq_len", "lr_schedule", "decay_iters", "min_lr", "warmup_iters", "seed"]
        + ["tied_embeddings"]
    ]


# porian_{purpose}_{schedule}_{dataset}_n{rows} -- see
# data/raw/porian/README.md for what each part means. Maps to (raw "hparams"
# value, raw "dataset" value, raw "decay" value or None to skip that filter).
PORIAN_RAW_DIR = "porian"
PORIAN_VARIANTS = {
    "porian_main_stable_rw_n566": ("sweep", "rw", None),
    "porian_chinchilla_cosine_decay_rw_n100": ("base", "rw", "chinchilla"),
    "porian_chinchilla_cosine_decay_owt2_n100": ("base", "owt2", "chinchilla"),
    "porian_kaplan_cosine_decay_rw_n32": ("base", "rw", "kaplan"),
    "porian_kaplan_cosine_decay_owt2_n32": ("base", "owt2", "kaplan"),
    "porian_seed_stable_rw_n42": ("seed", "rw", None),
    "porian_seed_stable_owt2_n42": ("seed", "owt2", None),
    "porian_tuned_stable_rw_n32": ("tuned", "rw", None),
    "porian_tuned_stable_owt2_n29": ("tuned", "owt2", None),
}


def _porian_adapter(hparams: str, dataset: str, decay: str | None):
    return partial(adapt_porian, hparams=hparams, dataset=dataset, decay=decay)


def adapt_gemstones(
    df: pd.DataFrame,
    lr_base: float = 5.0,
    decay_phase: str | None = None,
    is_gemstones_run: bool = True,
) -> pd.DataFrame:
    # Constants from the paper (arXiv:2502.06857) Sec. 3 / Appendix A.1:
    # context length 2048, world batch size 2048 sequences (~4M tokens/step);
    # lr follows a hybrid muP-style rule lr = lr_base / (width * sqrt(depth)),
    # constant after an 80M-token linear warmup -- lr_base=5 for the main
    # runs, 2.5 for the LR-ablation sweep (see GEMSTONE_VARIANTS below);
    # decay runs train for a further 10% of the tokens seen so far under a
    # linear LR-annealing schedule to 0. None of this applies to the
    # "external" Chinchilla data.
    df = df.rename(columns={"params_active_precise": "N", "tokens": "D", "final_loss": "loss"})
    if is_gemstones_run:
        df["gbsz"] = 2048
        df["seq_len"] = 2048
        df["lr"] = lr_base / (df["width"] * df["depth"] ** 0.5)
        df["warmup_iters"] = math.ceil(80_000_000 / (2048 * 2048))
        if decay_phase == "decay":
            df["decay_iters"] = df["step"] - df["step"] / 1.1
        elif decay_phase == "predecay":
            df["decay_iters"] = 0  # measured before any decay step ran
        else:
            df["decay_iters"] = None
        df["tied_embeddings"] = False  # Appendix A
        # N = params_active_precise = N_total see data/raw/gemstones/README.md).
        df["N_definition"] = "N_total"
    else:
        df["width"] = None
        df["depth"] = None
        df["gbsz"] = None
        df["seq_len"] = None
        df["lr"] = None
        df["warmup_iters"] = None
        df["decay_iters"] = None
        df["tied_embeddings"] = None
        df["N_definition"] = None
    return df[
        COLUMNS
        + ["width", "depth", "seq_len", "warmup_iters", "decay_iters"]
        + ["tied_embeddings", "N_definition"]
    ]


# Name mapping conventions for the Gemstones datasets.
# gemstone_{stable|decay|external}[_token range]_n{checkpoint rows}[_axis],
# where "stable" = constant LR schedule, "decay" = loss after the LR is
# linearly annealed to 0, and "external" = data pulled from Epoch AI's
# Chinchilla replication, redistributed by the paper's repo for comparison
# (not a Gemstones run). [_axis] is a trailing qualifier -- "lrablation"
# (half-LR sweep), "512width" (width=512 only), "predecay" (pre-branch
# checkpoint), "sampled_like_chinchilla" (downsampled to Chinchilla's (N, D)
# density) -- see data/raw/gemstones/README.md for more info.
GEMSTONE_RAW_DIR = "gemstones"
GEMSTONE_VARIANTS = {
    "gemstone_stable_100b_n220": "wandb_df_for_fitting_hot.jsonl",
    "gemstone_stable_100b_n1100": "wandb_df_for_fitting_hot_477.jsonl",
    "gemstone_stable_350b_n770": "wandb_df_for_fitting_hot_100b+.jsonl",
    "gemstone_stable_350b_n3850": "wandb_df_for_fitting_hot_100b+_477.jsonl",
    "gemstone_stable_120to350b_n528": "wandb_df_for_fitting_hot_120b+_only.jsonl",
    "gemstone_stable_120to350b_n2552": "wandb_df_for_fitting_hot_120b+_only_477.jsonl",
    "gemstone_stable_350b_n175_512width": "wandb_df_for_fitting_hot_100b+_512x.jsonl",
    "gemstone_stable_100b_n199_predecay": "wandb_df_for_fitting_cool_begin.jsonl",
    "gemstone_decay_11to110b_n199": "wandb_df_for_fitting_cool_end.jsonl",
    "gemstone_stable_100b_n200_lrablation": "wandb_df_for_fitting_lr_ablation_hot.jsonl",
    "gemstone_stable_100b_n1000_lrablation": "wandb_df_for_fitting_lr_ablation_hot_477.jsonl",
    "gemstone_stable_n24_sampled_like_chinchilla": "wandb_df_for_fitting_hot_sampled_like_chinchilla.jsonl",
    "gemstone_stable_n23_lrablation_sampled_like_chinchilla": "wandb_df_for_fitting_lr_ablation_hot_sampled_like_chinchilla.jsonl",
    "gemstone_external_chinchilla_n28": "wandb_df_for_fitting_slim_chinchilla.jsonl",
    "gemstone_external_chinchilla_n27_lrablation": "wandb_df_for_fitting_lr_ablation_slim_chinchilla.jsonl",
}


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


def adapt_warmstart(df: pd.DataFrame, method: str) -> pd.DataFrame:
    # Per-step logs (~2.5M rows), one row per eval. `post_hoc_df` is each
    # run's own output-file path, so grouping by it and keeping the row with
    # the largest `tokens` that has a logged `Validation Loss` gives that
    # run's final validation loss (a couple of runs are missing their true
    # final eval, so this isn't always literally the last step; some raw
    # rows have no `post_hoc_df`/`Validation Loss` at all -- groupby drops
    # them by default).
    #
    # The raw `method` column bundles 3 initialization techniques compared
    # on the same (N, D) grid: "mup" (Scratch, muP-transferred
    # hyperparameters, no inherited weights), "net2net" (Net2Net growth),
    # "paws" (the paper's Shrink-Zeros-Perturb/SZP method, Eq. 2 of
    # arXiv:2605.13405). Each has a different loss(N, D) surface, so each
    # gets its own output dataset (WARMSTART_VARIANTS below).
    #
    # N = target_N (the post-growth size evaluated); `base_N` (the
    # pre-growth size warmstarted from -- also the scale the run's
    # muP-transferred hyperparameters came from) is kept as an extra column.
    df = df.dropna(subset=["Validation Loss"]).sort_values("tokens")
    df = df.groupby("post_hoc_df", as_index=False).tail(1)
    df = df[df["method"] == method]
    df = df.rename(columns={
        "target_N": "N",
        "tokens": "D",
        "max_lr": "lr",
        "Validation Loss": "loss",
        "n_embd": "width",
        "n_layer": "depth",
        "block_size": "seq_len",
    })
    df["gbsz"] = df["micro_batch_size"] * df["accumulation_iters"] * df["devices"]
    # N (target_N/base_N) matches width/depth exactly via
    # 12*depth*width**2 + 101482*width (attention 4*width**2 + a 4x-ratio MLP
    # 8*width**2, no bias) across all 6 architectures in the paper's Table 5.
    # The 101482*width remainder mixes embedding/head/positional params the
    # paper doesn't break down, so tied_embeddings remains unknown.
    df["N_definition"] = "N_total"
    df["tied_embeddings"] = None
    # Appendix D.2: warmup-stable-decay with a linear cooldown to 0, warmup/
    # cooldown fixed at 1%/20% of each run's own total steps
    df["lr_schedule"] = "linear_decay"
    total_steps = df["D"] / (df["seq_len"] * df["gbsz"])
    df["warmup_iters"] = (df["warmup_fraction"] * total_steps).apply(round)
    df["decay_iters"] = (df["cooldown_fraction"] * total_steps).apply(round)
    return df[
        COLUMNS
        + ["width", "depth", "tied_embeddings", "seq_len", "N_definition"]
        + ["lr_schedule", "warmup_iters", "decay_iters", "min_lr"]
        + ["base_N"]
    ]


# warmstart_{method}_n{rows} -- see data/raw/warmstart/README.md. 
WARMSTART_RAW_DIR = "warmstart"
WARMSTART_VARIANTS = {
    "warmstart_scratch_n33": "mup",
    "warmstart_net2net_n27": "net2net",
    "warmstart_szp_n110": "paws",
}


def _warmstart_adapter(method: str):
    return partial(adapt_warmstart, method=method)


# Picks lr_base/decay_phase/is_gemstones_run for a GEMSTONE_VARIANTS name.
def _gemstone_adapter(name: str):
    lr_base = 2.5 if "_lrablation" in name else 5.0
    if "_predecay" in name:
        decay_phase = "predecay"
    elif name.startswith("gemstone_decay"):
        decay_phase = "decay"
    else:
        decay_phase = None
    is_gemstones_run = not name.startswith("gemstone_external")
    return partial(
        adapt_gemstones,
        lr_base=lr_base,
        decay_phase=decay_phase,
        is_gemstones_run=is_gemstones_run,
    )


STEPLAW_RAW_DIRS = {
    "steplaw_dense_n1911": "steplaw",
    "steplaw_moe_n708": "steplaw",
}

# name -> (adapter, raw filename(s), file format). filename(s) is a list for
# sources fetched as multiple files (e.g. HuggingFace train/test splits) that
# get concatenated before adapting.
ADAPTERS = {
    "oellm_english": (adapt_oellm_english, "all_points_clean.csv", "csv"),
    "oellm_multilingual": (adapt_oellm_multilingual, "decay_points_clean.csv", "csv"),
    "steplaw_dense_n1911": (adapt_steplaw_dense, "dense_lr_bs_loss.csv", "csv"),
    "steplaw_moe_n708": (adapt_steplaw_moe, "moe_lr_bs_loss.csv", "csv"),
    **{
        name: (_porian_adapter(hparams, dataset, decay), "experiment_results.pickle.xz", "pickle")
        for name, (hparams, dataset, decay) in PORIAN_VARIANTS.items()
    },
    **{
        name: (_gemstone_adapter(name), filename, "jsonl")
        for name, filename in GEMSTONE_VARIANTS.items()
    },
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
    **{
        name: (_warmstart_adapter(method), "warmstart_runs_flattened.parquet", "parquet")
        for name, method in WARMSTART_VARIANTS.items()
    },
}

READERS = {
    "csv": pd.read_csv,
    "pickle": pd.read_pickle,
    "jsonl": lambda path: pd.read_json(path, lines=True),
    "parquet": pd.read_parquet,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--extended",
        action="store_true",
        help="Also write extra columns beyond the common "
        f"{COLUMNS} schema (e.g. width/depth/warmup iters, etc.).",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=list(ADAPTERS),
        default=None,
        help="Only preprocess these sources (default: all sources).",
    )
    args = parser.parse_args()
    sources = args.sources or list(ADAPTERS)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in sources:
        adapt, filenames, filetype = ADAPTERS[name]

        if name in GEMSTONE_VARIANTS:
            raw_dir = RAW_DIR / GEMSTONE_RAW_DIR
        elif name in STEPLAW_RAW_DIRS:
            raw_dir = RAW_DIR / STEPLAW_RAW_DIRS[name]
        elif name in PORIAN_VARIANTS:
            raw_dir = RAW_DIR / PORIAN_RAW_DIR
        elif name in WARMSTART_VARIANTS:
            raw_dir = RAW_DIR / WARMSTART_RAW_DIR
        else:
            raw_dir = RAW_DIR / name
        try:
            if isinstance(filenames, list):
                df = pd.concat(
                    [READERS[filetype](raw_dir / f) for f in filenames],
                    ignore_index=True,
                )
            else:
                df = READERS[filetype](raw_dir / filenames)
        except FileNotFoundError:
            print(f"{name}: raw data not found under {raw_dir}, skipping")
            continue
        out = adapt(df)
        if not args.extended:
            out = out[COLUMNS]
        # some sources store N/D as int64 (e.g. steplaw); 6*N*D can overflow int64
        # at large scales and silently wrap negative, so enforce float64 here.
        # (N_definition is a string label, not a numeric N-variant -- skip it.)
        float_cols = [
            c for c in out.columns if c == "D" or (c.startswith("N") and c != "N_definition")
        ]
        out = out.astype({c: "float64" for c in float_cols})
        out_path = OUT_DIR / f"{name}.csv"
        out.to_csv(out_path, index=False)
        print(f"{name}: {len(out)} rows -> {out_path}")


if __name__ == "__main__":
    main()
