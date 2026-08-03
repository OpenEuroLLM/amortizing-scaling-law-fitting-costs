"""Transform each raw source CSV into the common N, D, lr, gbsz, loss schema
that iterative_fits_L0_form.py and model_based_fits.py expect as --input.
"""
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
OUT_DIR = Path(__file__).parent.parent / "data" / "preprocessed"

COLUMNS = ["N", "D", "lr", "gbsz", "loss"]

# (depth, width) -> parameter count in millions, from Table 2, Appendix B of
# Porian et al. (arXiv:2406.19146). Their table also lists N_exact/N_eff/N_Kaplan
# variants; this plain "N" column is the one used, matching the N_Porian_exact
# convention already used for oellm_multilingual.
PORIAN_PARAM_LOOKUP_M = {
    (3, 96): 5.173,
    (4, 128): 7.504,
    (5, 160): 9.810,
    (6, 224): 15.60,
    (8, 288): 22.49,
    (9, 320): 28.67,
    (10, 384): 37.06,
    (12, 480): 57.38,
    (14, 576): 84.79,
    (15, 640): 108.5,
    (18, 704): 149.0,
    (21, 832): 220.9,
    (23, 1024): 347.1,
    (26, 1120): 455.3,
    (26, 1312): 612.0,
    (30, 1504): 901.7,
}


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
    df["N"] = [
        PORIAN_PARAM_LOOKUP_M[(depth, width)] * 1e6
        for depth, width in zip(df["depth"], df["width"])
    ]
    return df[COLUMNS]


def adapt_gemstones(df: pd.DataFrame) -> pd.DataFrame:
    # No lr/gbsz anywhere in this source (main sweep uses a muP-derived LR
    # rule that isn't logged per-run) -- left blank per-request; usable for
    # envelope fits, not for the surrogate scripts that need all 4 features.
    df = df.rename(columns={"params_active_precise": "N", "tokens": "D", "final_loss": "loss"})
    df["lr"] = None
    df["gbsz"] = None
    return df[COLUMNS]


# name -> (adapter, raw filename, file format)
ADAPTERS = {
    "oellm_english": (adapt_oellm_english, "all_points_clean.csv", "csv"),
    "oellm_multilingual": (adapt_oellm_multilingual, "decay_points_clean.csv", "csv"),
    "steplaw": (adapt_steplaw, "dense_lr_bs_loss.csv", "csv"),
    "porian": (adapt_porian, "experiment_results.pickle.xz", "pickle"),
    "gemstones": (adapt_gemstones, "wandb_df_for_fitting_hot.jsonl", "jsonl"),
}

READERS = {
    "csv": pd.read_csv,
    "pickle": pd.read_pickle,
    "jsonl": lambda path: pd.read_json(path, lines=True),
}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, (adapt, filename, filetype) in ADAPTERS.items():
        df = READERS[filetype](RAW_DIR / name / filename)
        out = adapt(df)
        # some sources store N/D as int64 (e.g. steplaw); 6*N*D can overflow int64
        # at large scales and silently wrap negative, so enforce float64 here.
        out = out.astype({"N": "float64", "D": "float64"})
        out_path = OUT_DIR / f"{name}.csv"
        out.to_csv(out_path, index=False)
        print(f"{name}: {len(out)} rows -> {out_path}")


if __name__ == "__main__":
    main()
