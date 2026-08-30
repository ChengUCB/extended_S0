"""Shared constants and input validation for the paracetamol GP notebooks."""

from __future__ import annotations

from pathlib import Path


TEMPERATURE_K = 303.15
K_B_EV_PER_K = 8.617333262145e-5
KBT = K_B_EV_PER_K * TEMPERATURE_K
KCAL_2_EV = 0.0433641153087705

# ``szero.prepare_gp_gradient_data`` uses component names in its mapping keys.
# The CSV site labels are N=paracetamol, O=water, and C=ethanol.
COLUMN_MAP = {
    "x_Para": "x_para",
    "x_Water": "x_water",
    "x_EtOH": "x_ethanol",
    "S_ParaPara": "S0_NN",
    "S_ParaWater": "S0_NO",
    "S_ParaEtOH": "S0_NC",
    "S_WaterWater": "S0_OO",
    "S_WaterEtOH": "S0_OC",
    "S_EtOHEtOH": "S0_CC",
}

DATA_FILE = Path(__file__).with_name("S0_paracetamol_water_ethanol_303K.csv")
PAIR_COLUMNS = ("NN", "OO", "CC", "NO", "NC", "OC")
REQUIRED_COLUMNS = {
    "composition_key",
    "n_para",
    "n_water",
    "n_ethanol",
    "x_para",
    "x_water",
    "x_ethanol",
    *(f"S0_{pair}" for pair in PAIR_COLUMNS),
    *(f"S0_{pair}_split{split}" for split in (1, 2, 3) for pair in PAIR_COLUMNS),
}


def load_s0(path: str | Path | None = None):
    """Load the bundled S(0) table and validate the analysis input schema."""
    import pandas as pd

    csv_path = DATA_FILE if path is None else Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Paracetamol S(0) input is missing: {csv_path}. "
            "Use the bundled S0_paracetamol_water_ethanol_303K.csv or pass its path explicitly."
        )

    frame = pd.read_csv(csv_path, comment="#")
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(
            f"{csv_path} does not satisfy the paracetamol S(0) schema; "
            f"missing columns: {', '.join(missing)}"
        )
    if frame.empty:
        raise ValueError(f"{csv_path} contains no compositions")
    if frame["composition_key"].duplicated().any():
        raise ValueError(f"{csv_path} contains duplicate composition_key values")
    return frame
