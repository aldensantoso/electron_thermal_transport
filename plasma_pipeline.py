import os
import itertools
import numpy as np
import pandas as pd
from scipy.io import readsav
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LassoCV
from sklearn.metrics import r2_score

PROJECT_DIR = r'/Users/aldensantoso/Documents/plasma_research/ML Project/src/plasma_pipeline.py'
RAW_SAV_PATH = os.path.join(PROJECT_DIR, "data", "APRIL21.SAV")
MERGED_CSV_PATH = os.path.join(PROJECT_DIR, "combined_plasma_data_clean.csv")
FINAL_CLEANED_PATH = os.path.join(PROJECT_DIR, "combined_plasma_data_clean_clean.csv")


def norm_name(x):
    if x is None:
        return ""
    return str(x).strip().lower().replace("\ufeff", "").replace(" ", "").replace("_", "").replace("-", "")


def find_col(df, alias_groups):
    norm_map = {norm_name(c): c for c in df.columns}

    for group in alias_groups:
        if isinstance(group, str):
            group = [group]
        for alias in group:
            key = norm_name(alias)
            if key in norm_map:
                return norm_map[key]

    for group in alias_groups:
        if isinstance(group, str):
            group = [group]
        for alias in group:
            alias_key = norm_name(alias)
            for col in df.columns:
                col_key = norm_name(col)
                if alias_key in col_key or col_key in alias_key:
                    return col
    return None


def load_and_merge_sav(raw_sav_path=None):
    if raw_sav_path is None:
        raw_sav_path = RAW_SAV_PATH

    if not os.path.exists(raw_sav_path):
        raise FileNotFoundError(raw_sav_path)

    data = readsav(raw_sav_path, verbose=False)

    tr_struct = data["tr"][0]
    tr_dict = {name: tr_struct[name] for name in tr_struct.dtype.names}
    tr_df = pd.DataFrame(tr_dict)

    mp_struct = data["mp"][0]
    mp_dict = {name: mp_struct[name] for name in mp_struct.dtype.names}
    mp_df = pd.DataFrame(mp_dict)

    common_columns = set(mp_df.columns).intersection(tr_df.columns)
    shot_key = None
    for col in common_columns:
        if "shot" in col.lower():
            shot_key = col
            break

    if shot_key:
        combined_df = pd.merge(mp_df, tr_df, on=shot_key, how="inner")
    else:
        if len(mp_df) != len(tr_df):
            raise ValueError(
                f"No common shot key and row counts differ "
                f"({len(mp_df)} vs {len(tr_df)})."
            )
        overlap = set(mp_df.columns).intersection(tr_df.columns)
        if overlap:
            mp_df = mp_df.rename(columns={c: f"mp_{c}" for c in overlap})
            tr_df = tr_df.rename(columns={c: f"tr_{c}" for c in overlap})
        combined_df = pd.concat(
            [mp_df.reset_index(drop=True), tr_df.reset_index(drop=True)],
            axis=1,
            sort=False,
        )

    combined_df.to_csv(MERGED_CSV_PATH, index=False)
    return combined_df


def clean_combined_csv(in_csv=None, out_csv=None):
    if in_csv is None:
        in_csv = MERGED_CSV_PATH
    if out_csv is None:
        out_csv = FINAL_CLEANED_PATH

    df = pd.read_csv(in_csv)
    df.columns = df.columns.str.strip().str.replace("\ufeff", "")

    df = df.drop(columns=[c for c in ["v_A_proxy", "f_norm", "n_norm", "dB2"] if c in df.columns])

    etau_col = find_col(df, [["ETAU", "TAUE", "TAU_E", "ETAU_E"]])
    nes_col = find_col(df, [["NES", "NE", "N_E", "NEE", "DENSITY"]])

    if etau_col is None:
        raise KeyError("ETAU / tau_e column not found in CSV")
    if nes_col is None:
        raise KeyError("Electron density column (NES / NE) not found in CSV")

    avgmodef_col = find_col(df, [["AVGMODEF", "AVG_MODEF"]])
    avgmodef2_col = find_col(df, [["AVGMODEF2", "AVG_MODEF2"]])
    bzxr_col = find_col(df, [["BZXR", "B_ZXR"]])
    te_col = find_col(df, [["TE", "T_E"]])
    pvol_col = find_col(df, [["PVOL", "P_VOL"]])
    raxis_col = find_col(df, [["RAXIS", "R_AXIS"]])
    tex_col = find_col(df, [["TEX", "T_EX"]])

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)

    df = df[df[etau_col] > 0].copy()
    df = df[df[nes_col] > 0].copy()

    for col in [avgmodef_col, avgmodef2_col, bzxr_col, te_col, pvol_col, raxis_col, tex_col]:
        if col is not None:
            df = df[df[col] > 0].copy()

    df = df.dropna(axis=0, how="any").reset_index(drop=True)
    df.to_csv(out_csv, index=False)
    return df


def load_and_clean_data(filepath=None):
    if filepath is None:
        filepath = FINAL_CLEANED_PATH

    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip().str.replace("\ufeff", "")

    rename_dict = {
        "PBEAM_TOT": "P_beam",
        "BZXR": "B_T",
        "IBEAM_A": "I_P",
        "NES": "n_e",
        "AVGMODEF": "f",
        "AVGMODEN": "n",
        "TOTPWR": "dB",
        "ETAU": "tau_e",
    }

    for old, new in rename_dict.items():
        if old in df.columns:
            df = df.rename(columns={old: new})

    required = ["P_beam", "B_T", "I_P", "n_e", "f", "n", "dB", "tau_e"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    df = df[(df["tau_e"] > 0) & (df["n_e"] > 0)].copy()

    numeric_df = df[required].copy()
    varying_cols = [c for c in required if numeric_df[c].nunique() > 1]
    if varying_cols:
        z_scores = np.abs(stats.zscore(numeric_df[varying_cols]))
        df = df[(z_scores < 3).all(axis=1)].copy()

    return df


def engineer_features(df):
    df = df.copy()
    required = ["P_beam", "B_T", "I_P", "n_e", "f", "n", "dB", "tau_e"]
    for c in required:
        if c not in df.columns:
            raise KeyError(f"Missing feature column: {c}")

    df = df[df["n_e"] > 0].copy()
    df["v_A_proxy"] = df["B_T"] / np.sqrt(df["n_e"])
    df["f_norm"] = df["f"] / df["v_A_proxy"]
    df["n_norm"] = df["n"] / df["v_A_proxy"]
    df["dB2"] = df["dB"] ** 2

    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df, ["P_beam", "B_T", "I_P", "n_e"], ["f_norm", "n_norm", "dB2"], "tau_e"


def train_pipeline(df, X_control_cols, Y_wave_cols, Y_target_col):
    X1 = df[X_control_cols]
    Y1 = df[Y_wave_cols]
    X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1, Y1, test_size=0.2, random_state=42)

    scaler_X1 = StandardScaler()
    X1_train_scaled = scaler_X1.fit_transform(X1_train)
    X1_test_scaled = scaler_X1.transform(X1_test)

    wave_model = MultiOutputRegressor(RandomForestRegressor(n_estimators=150, random_state=42))
    wave_model.fit(X1_train_scaled, Y1_train)
    Y1_pred = wave_model.predict(X1_test_scaled)
    print(f"Wave R2: {r2_score(Y1_test, Y1_pred):.3f}")

    X2 = pd.concat([df[X_control_cols], df[Y_wave_cols]], axis=1)
    Y2 = df[Y_target_col].values.ravel()
    X2_train, X2_test, Y2_train, Y2_test = train_test_split(X2, Y2, test_size=0.2, random_state=42)

    scaler_X2 = StandardScaler()
    X2_train_scaled = scaler_X2.fit_transform(X2_train)
    X2_test_scaled = scaler_X2.transform(X2_test)

    confinement_model = LassoCV(cv=5, random_state=42)
    confinement_model.fit(X2_train_scaled, Y2_train)
    Y2_pred = confinement_model.predict(X2_test_scaled)
    print(f"Confinement R2: {r2_score(Y2_test, Y2_pred):.3f}")

    return wave_model, confinement_model, scaler_X1, scaler_X2


def optimize_operating_space(wave_model, confinement_model, scaler_X1, scaler_X2, X_control_cols):
    P_beam_grid = np.linspace(4.0, 7.0, 10)
    B_T_grid = np.linspace(0.3, 0.55, 10)
    I_P_grid = np.linspace(0.7, 1.3, 5)
    n_e_grid = np.linspace(2.0, 5.0, 5)

    grid = list(itertools.product(P_beam_grid, B_T_grid, I_P_grid, n_e_grid))
    df_grid = pd.DataFrame(grid, columns=X_control_cols)

    pred_wave = wave_model.predict(scaler_X1.transform(df_grid))
    df_grid["f_norm"] = pred_wave[:, 0]
    df_grid["n_norm"] = pred_wave[:, 1]
    df_grid["dB2"] = pred_wave[:, 2]

    X2_grid = df_grid[X_control_cols + ["f_norm", "n_norm", "dB2"]]
    df_grid["predicted_tau_e"] = confinement_model.predict(scaler_X2.transform(X2_grid))

    high_power = df_grid[df_grid["P_beam"] >= 6.0].sort_values(by="predicted_tau_e", ascending=False)
    print("\nTop 5 configurations:")
    print(high_power.head(5).to_string(index=False))
    return high_power.head(5)