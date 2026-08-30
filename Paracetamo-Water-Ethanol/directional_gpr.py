"""Directional-gradient preparation used by the paracetamol production notebook.

The three composition regions and their tangent directions are the same ones
demonstrated in ``test_nonstationary_GP.ipynb``; this module makes that existing
workflow importable by ``Para-water-eth_Final.ipynb``.
"""

from __future__ import annotations

from collections.abc import Mapping


REQUIRED_MAP_KEYS = {
    "x_Para",
    "x_Water",
    "x_EtOH",
    "S_ParaPara",
    "S_ParaWater",
    "S_ParaEtOH",
    "S_WaterWater",
    "S_WaterEtOH",
    "S_EtOHEtOH",
}


def prepare_para_directional_data(frame, column_map: Mapping[str, str], kb_t: float):
    """Return GP points, gradient values, and directions in (x_para, x_ethanol)."""
    import numpy as np

    try:
        from szero import prepare_gp_gradient_data
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "The 'szero' package is required. Clone https://github.com/ChengUCB/S0_multi "
            "next to extended_S0 or set SZERO_DIR as documented in REPRODUCIBILITY.md."
        ) from error

    if not isinstance(column_map, Mapping):
        raise TypeError("column_map must be a mapping")
    missing_map = sorted(REQUIRED_MAP_KEYS - set(column_map))
    if missing_map:
        raise ValueError("column_map is missing keys: " + ", ".join(missing_map))
    missing_columns = sorted(set(column_map.values()) - set(frame.columns))
    if missing_columns:
        raise ValueError("S(0) frame is missing columns: " + ", ".join(missing_columns))
    if frame.empty:
        raise ValueError("S(0) frame contains no compositions")
    if not isinstance(kb_t, (int, float)) or kb_t <= 0:
        raise ValueError("kb_t must be a positive scalar in the desired gradient energy unit")

    x_para = column_map["x_Para"]
    x_water = column_map["x_Water"]
    x_ethanol = column_map["x_EtOH"]
    segments = (
        (
            ("Para", "Water", "EtOH"),
            ("Para", "EtOH"),
            ((1.0, 0.0), (0.0, 1.0)),
            (frame[x_water] > 1e-9) & (frame[x_ethanol] > 1e-9),
        ),
        (
            ("Para", "Water"),
            ("Para",),
            ((1.0, 0.0),),
            frame[x_ethanol] <= 1e-9,
        ),
        (
            ("Para", "EtOH"),
            ("Para",),
            ((1.0, -1.0),),
            frame[x_water] <= 1e-9,
        ),
    )

    parts = []
    for components, independent, tangents, mask in segments:
        subset = frame.loc[mask].reset_index(drop=True)
        if subset.empty:
            continue
        local_map = {f"x_{component}": column_map[f"x_{component}"] for component in components}
        local_map.update(
            {
                f"S_{first}{second}": column_map[f"S_{first}{second}"]
                for index, first in enumerate(components)
                for second in components[index:]
            }
        )
        _, gradients, _ = prepare_gp_gradient_data(
            subset,
            independent_components=list(independent),
            column_map=local_map,
            components=list(components),
            species="Para",
            excess=True,
            kb_t=float(kb_t),
            as_torch=False,
        )
        points = subset[[x_para, x_ethanol]].to_numpy(float)
        parts.append(
            (
                np.repeat(points, len(tangents), axis=0),
                np.asarray(gradients, dtype=float).reshape(-1),
                np.tile(np.asarray(tangents, dtype=float), (len(subset), 1)),
            )
        )

    if not parts:
        raise ValueError("S(0) frame has no supported ternary or binary-edge compositions")
    return (
        np.vstack([part[0] for part in parts]),
        np.concatenate([part[1] for part in parts]),
        np.vstack([part[2] for part in parts]),
    )
