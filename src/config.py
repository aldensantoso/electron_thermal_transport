from pathlib import Path

PROJECT_ROOT = Path("/Users/aldensantoso/Documents/plasma_research/ML Project").resolve()


SAV_PATH = PROJECT_ROOT / "data" / "APRIL21.SAV"
CSV_PATH = PROJECT_ROOT / "combined_plasma_data_clean_clean.csv"

COLUMN_MAPPING = {
    "PBEAM_TOT": "P_beam",
    "PBEAM_A": "P_beam",
    "PBEAM_B": "P_beam",
    "PBEAM_C": "P_beam",

    "mp_BZXR": "B_T",
    "tr_BZXR": "B_T",
    "BZXR": "B_T",
    "BPTE": "B_T",
    "mp_BPTE": "B_T",
    "tr_BPTE": "B_T",

    "IBEAM_A": "I_P",
    "IBEAM_B": "I_P",
    "IBEAM_C": "I_P",

    "mp_NES": "n_e",
    "tr_NES": "n_e",
    "NES": "n_e",
    "NEUTT": "n_e",
    "mp_NEUTT": "n_e",
    "tr_NEUTT": "n_e",

    "AVGMODEF": "f",
    "AVGMODEF2": "f",
    "AVGMODEFNZ": "f",
    "AVGMODEF2NZ": "f",
    "AVGMODEN": "n",
    "AVGMODEN2": "n",
    "AVGMODENNZ": "n",
    "AVGMODEN2NZ": "n",

    "TOTPWR": "dB",
    "TOTPWRLF": "dB",
    "TOTPWRNZ": "dB",

    "ETAU": "tau_e",
    "THTAU": "tau_e",
    "TOTAU": "tau_e",

    "mp_TE": "TE",
    "tr_TE": "TE",
    "TE": "TE",
    "PVOL": "PVOL",
    "RAXIS": "RAXIS",
    "TEX": "TEX",
}