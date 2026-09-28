import os
import pandas as pd
import numpy as np
from scipy.io import readsav

def extract_sav_to_csv(sav_path, out_dir=None):
    if out_dir is None:
        out_dir = os.getcwd()

    data = readsav(sav_path, verbose=False)
    tr_struct = data["tr"][0]
    tr_dict = {name: tr_struct[name] for name in tr_struct.dtype.names}
    tr_df = pd.DataFrame(tr_dict)
    tr_csv = os.path.join(out_dir, "tr_df.csv")
    tr_df.to_csv(tr_csv, index=False)

    mp_struct = data["mp"][0]
    mp_dict = {name: mp_struct[name] for name in mp_struct.dtype.names}
    mp_df = pd.DataFrame(mp_dict)
    mp_csv = os.path.join(out_dir, "mp_df.csv")
    mp_df.to_csv(mp_csv, index=False)

    return tr_csv, mp_csv


def merge_mp_tr(mp_filepath, tr_filepath, destination_path=None):
    if destination_path is None:
        destination_path = os.path.join(
            os.path.dirname(mp_filepath), "combined_plasma_data_clean.csv"
        )

    mp_df = pd.read_csv(mp_filepath)
    tr_df = pd.read_csv(tr_filepath)

    mp_cols = set(mp_df.columns)
    tr_cols = set(tr_df.columns)
    common_columns = mp_cols.intersection(tr_cols)

    shot_key = None
    for col in common_columns:
        if "shot" in col.lower():
            shot_key = col
            break

    if shot_key:
        combined_df = pd.merge(mp_df, tr_df, on=shot_key, how="inner")
        print(f"Successfully merged using alignment key: '{shot_key}'")
    else:
        combined_df = pd.concat([mp_df, tr_df], axis=1)
        combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]
        print("No common shot key found. Concatenated datasets side-by-side.")

    combined_df.to_csv(destination_path, index=False)
    print(f"mp dimensions {mp_df.shape}")
    print(f"tr dimensions {tr_df.shape}")
    print(f"\n[SUCCESS] New matrix created with dimensions: {combined_df.shape}")
    print(f"Saved successfully to your machine at:\n--> {destination_path}")

    return combined_df


def clean_combined_csv(in_csv=None, out_csv=None):
    if in_csv is None:
        in_csv = (
            "/Users/aldensantoso/Documents/plasma_research/ML"
            " Project/combined_plasma_data_clean.csv"
        )
    if out_csv is None:
        out_csv = in_csv.replace(".csv", "_clean.csv")

    if not os.path.exists(in_csv):
        raise FileNotFoundError(in_csv)

    df = pd.read_csv(in_csv)
    df.columns = df.columns.str.strip().str.replace("\ufeff", "")

    derived_to_remove = ["v_A_proxy", "f_norm", "n_norm", "dB2"]
    df = df.drop(columns=[c for c in derived_to_remove if c in df.columns])

    etau_keys = ("ETAU",)
    nes_keys = ("NES",)
    avgmodef_keys = ("AVGMODEF",)
    avgmodef2_keys = ("AVGMODEF2",)
    bzxr_keys = ("BZXR",)
    te_keys = ("TE",)
    pvol_keys = ("PVOL",)
    raxis_keys = ("RAXIS",)
    tex_keys = ("TEX",)

    etau_col = next((c for c in etau_keys if c in df.columns), None)
    nes_col = next((c for c in nes_keys if c in df.columns), None)
    avgmodef_col = next((c for c in avgmodef_keys if c in df.columns), None)
    avgmodef2_col = next((c for c in avgmodef2_keys if c in df.columns), None)
    bzxr_col = next((c for c in bzxr_keys if c in df.columns), None)
    te_col = next((c for c in te_keys if c in df.columns), None)
    pvol_col = next((c for c in pvol_keys if c in df.columns), None)
    raxis_col = next((c for c in raxis_keys if c in df.columns), None)
    tex_col = next((c for c in tex_keys if c in df.columns), None)

    if etau_col is None:
        etau_col = next(
            (c for c in df.columns if c.strip().lower() in ("etau", "tau_e", "taue")),
            None,
        )
    if etau_col is None:
        raise KeyError("ETAU / tau_e column not found in CSV")

    if nes_col is None:
        nes_col = next(
            (c for c in df.columns if c.strip().lower() in ("nes", "ne", "n_e")),
            None,
        )
    if nes_col is None:
        raise KeyError("Electron density column (NES / NE) not found in CSV")

    if avgmodef_col is None:
        avgmodef_col = next(
            (c for c in df.columns if c.strip().lower() == "avgmodef"),
            None,
        )
    if avgmodef2_col is None:
        avgmodef2_col = next(
            (c for c in df.columns if c.strip().lower() == "avgmodef2"),
            None,
        )
    if bzxr_col is None:
        bzxr_col = next(
            (c for c in df.columns if c.strip().lower() == "bzxr"),
            None,
        )
    if te_col is None:
        te_col = next(
            (c for c in df.columns if c.strip().lower() == "te"),
            None,
        )
    if pvol_col is None:
        pvol_col = next(
            (c for c in df.columns if c.strip().lower() == "pvol"),
            None,
        )
    if raxis_col is None:
        raxis_col = next(
            (c for c in df.columns if c.strip().lower() == "raxis"),
            None,
        )
    if tex_col is None:
        tex_col = next(
            (c for c in df.columns if c.strip().lower() == "tex"),
            None,
        )

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)

    df = df[df[etau_col] > 0].copy()
    df = df[df[nes_col] > 0].copy()
    df = df[df[avgmodef_col] > 0].copy()
    df = df[df[avgmodef2_col] > 0].copy()
    df = df[df[bzxr_col] > 0].copy()
    df = df[df[te_col] > 0].copy()
    df = df[df[pvol_col] > 0].copy()
    df = df[df[raxis_col] > 0].copy()
    df = df[df[tex_col] > 0].copy()

    df = df.dropna(axis=0, how="any").reset_index(drop=True)

    df.to_csv(out_csv, index=False)
    print(f"Saved cleaned CSV: {out_csv}  (rows remaining: {len(df)})")
    return df
