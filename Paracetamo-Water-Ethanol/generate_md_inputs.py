#!/usr/bin/env python3
"""Generate LAMMPS input-config.dat files from a composition manifest.

Preferred reproducible entry point:

    python3 generate_md_inputs.py --manifest composition.json

The manifest contract is documented by ``md_input_manifest.schema.json``.  The
inline ``level_composition`` mapping remains available for backwards-compatible
local use, but an empty mapping is an error instead of a successful no-op.
"""

from __future__ import annotations


# Keep this dictionary empty in the distributed script.
# Formatting example (all composition fields are required):
#
# level_composition = {
#     "level_001": {                    # key = output folder name
#         "n_para": 2,                   # paracetamol molecules
#         "n_water": 5,                  # water molecules
#         "n_ethanol": 3,                # ethanol molecules
#         "box_edge_angstrom": 25.0,     # cubic primitive-box edge in angstrom
#         "seed": 42,                    # fixed integer random seed
#     },
# }
#
# Counts are for the generated primitive cell. If a later LAMMPS input uses
# ``replicate 2 2 2``, final molecule counts are eight times these values.
level_composition = {}

import argparse
import math
import random
import sys
from collections import OrderedDict


# ---------------------------------------------------------------------------
# Unified type numbers (by CHARMM type name)
# ---------------------------------------------------------------------------
CHARMM_TYPE_TO_UNIFIED = {
    "H":   1,
    "HA2": 2,
    "HA3": 3,
    "HP":  4,
    "HT":  5,
    "C":   6,
    "CA":  7,
    "CT2": 8,
    "CT3": 9,
    "NH1": 10,
    "O":   11,
    "OH1": 12,
    "OT":  13,
}

# Inverse map: unified -> name
UNIFIED_TO_CHARMM_TYPE = {v: k for k, v in CHARMM_TYPE_TO_UNIFIED.items()}

# Masses
MASSES = {
    1:  1.008,   # H
    2:  1.008,   # HA2
    3:  1.008,   # HA3
    4:  1.008,   # HP
    5:  1.008,   # HT
    6:  12.011,  # C
    7:  12.011,  # CA
    8:  12.011,  # CT2
    9:  12.011,  # CT3
    10: 14.007,  # NH1
    11: 15.999,  # O
    12: 15.999,  # OH1
    13: 15.999,  # OT
}

# Pair Coeffs fallback: (epsilon, sigma). Production output is rebuilt from
# template data so CHARMM 1-4 LJ parameters are preserved.
PAIR_COEFFS = {
    1:  (0.04600000000000, 0.40001352444501),  # H
    2:  (0.03400000000000, 2.38760856461611),  # HA2
    3:  (0.02400000000000, 2.38760856461611),  # HA3
    4:  (0.03000000000000, 2.42003727795642),  # HP
    5:  (0.04600000000000, 0.40001352444501),  # HT
    6:  (0.11000000000000, 3.56359487256136),  # C
    7:  (0.07000000000000, 3.55005321204562),  # CA
    8:  (0.05600000000000, 3.58141284692416),  # CT2
    9:  (0.07800000000000, 3.63486677001258),  # CT3
    10: (0.20000000000000, 3.29632525711926),  # NH1
    11: (0.12000000000000, 3.02905564167715),  # O
    12: (0.15210000000000, 3.15378146221680),  # OH1
    13: (0.15210000000000, 3.15057422683150),  # OT
}


def parse_data_file(filepath):
    """Parse a LAMMPS data file and return its sections as structured data."""
    sections = {
        "header": [],
        "counts": None,
        "box": [],
        "masses": None,
        "pair_coeffs": None,
        "atoms": [],        # list of (atom_id, mol_id, type, charge, x, y, z, comment)
        "bond_coeffs": None,
        "bonds": [],        # list of (bond_type, atom1, atom2, comment)
        "angle_coeffs": None,
        "angles": [],       # list of (angle_type, a1, a2, a3, comment)
        "dihedral_coeffs": None,
        "dihedrals": [],    # list of (dih_type, a1, a2, a3, a4, comment)
        "improper_coeffs": None,
        "impropers": [],
    }

    with open(filepath, "r") as f:
        lines = f.readlines()

    i = 0
    # Parse header: first blank-line-separated block
    while i < len(lines):
        line = lines[i]
        if line.strip() == "":
            i += 1
            break
        sections["header"].append(line.rstrip())
        i += 1

    # Parse body by section keywords
    current_section = None
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        if line in ("Masses", "Pair Coeffs", "Atoms", "Bond Coeffs", "Bonds",
                     "Angle Coeffs", "Angles", "Dihedral Coeffs", "Dihedrals",
                     "Improper Coeffs", "Impropers"):
            current_section = line.lower().replace(" ", "_")
            if current_section == "masses":
                sections["masses"] = {}
            elif current_section == "pair_coeffs":
                sections["pair_coeffs"] = {}
            elif current_section == "bond_coeffs":
                sections["bond_coeffs"] = {}
            elif current_section == "angle_coeffs":
                sections["angle_coeffs"] = {}
            elif current_section == "dihedral_coeffs":
                sections["dihedral_coeffs"] = {}
            elif current_section == "improper_coeffs":
                sections["improper_coeffs"] = {}
            i += 1
            continue

        if current_section is None:
            # Detect box bounds: two numeric values followed by xlo/xhi etc.
            parts = line.split("#")[0].split()
            if len(parts) >= 2:
                try:
                    a = float(parts[0])
                    b = float(parts[1])
                    if len(parts) >= 3 and parts[2] in ("xlo", "ylo", "zlo"):
                        sections["box"].append((a, b))
                except ValueError:
                    pass
            i += 1
            continue

        # Parse based on current section
        if current_section == "masses":
            parts = line.split("#")[0].split()
            if len(parts) >= 2:
                sections["masses"][int(parts[0])] = float(parts[1])
        elif current_section == "pair_coeffs":
            parts = line.split("#")[0].split()
            if len(parts) >= 3:
                sections["pair_coeffs"][int(parts[0])] = tuple(
                    float(x) for x in parts[1:])
        elif current_section == "atoms":
            parts = line.split("#")
            data_part = parts[0].split()
            comment = parts[1].strip() if len(parts) > 1 else ""
            if len(data_part) >= 7:
                sections["atoms"].append((
                    int(data_part[0]), int(data_part[1]),
                    int(data_part[2]), float(data_part[3]),
                    float(data_part[4]), float(data_part[5]), float(data_part[6]),
                    comment))
        elif current_section == "bond_coeffs":
            parts = line.split("#")[0].split()
            if len(parts) >= 3:
                sections["bond_coeffs"][int(parts[0])] = (
                    float(parts[1]), float(parts[2]))
        elif current_section == "bonds":
            parts = line.split("#")
            data_part = parts[0].split()
            comment = parts[1].strip() if len(parts) > 1 else ""
            if len(data_part) == 3:
                # format: type atom1 atom2 (no ID)
                sections["bonds"].append((
                    int(data_part[0]), int(data_part[1]),
                    int(data_part[2]), comment))
            elif len(data_part) >= 4:
                # format: id type atom1 atom2
                sections["bonds"].append((
                    int(data_part[1]), int(data_part[2]),
                    int(data_part[3]), comment))
        elif current_section == "angle_coeffs":
            parts = line.split("#")[0].split()
            if len(parts) >= 3:
                sections["angle_coeffs"][int(parts[0])] = tuple(
                    float(x) for x in parts[1:])
        elif current_section == "angles":
            parts = line.split("#")
            data_part = parts[0].split()
            comment = parts[1].strip() if len(parts) > 1 else ""
            if len(data_part) == 4:
                sections["angles"].append((
                    int(data_part[0]), int(data_part[1]),
                    int(data_part[2]), int(data_part[3]), comment))
            elif len(data_part) >= 5:
                sections["angles"].append((
                    int(data_part[1]), int(data_part[2]),
                    int(data_part[3]), int(data_part[4]), comment))
        elif current_section == "dihedral_coeffs":
            parts = line.split("#")[0].split()
            if len(parts) >= 4:
                sections["dihedral_coeffs"][int(parts[0])] = tuple(
                    float(x) for x in parts[1:])
        elif current_section == "dihedrals":
            parts = line.split("#")
            data_part = parts[0].split()
            comment = parts[1].strip() if len(parts) > 1 else ""
            if len(data_part) == 5:
                sections["dihedrals"].append((
                    int(data_part[0]), int(data_part[1]),
                    int(data_part[2]), int(data_part[3]),
                    int(data_part[4]), comment))
            elif len(data_part) >= 6:
                sections["dihedrals"].append((
                    int(data_part[1]), int(data_part[2]),
                    int(data_part[3]), int(data_part[4]),
                    int(data_part[5]), comment))
        elif current_section == "improper_coeffs":
            parts = line.split("#")[0].split()
            if len(parts) >= 3:
                sections["improper_coeffs"][int(parts[0])] = tuple(
                    float(x) for x in parts[1:])
        elif current_section == "impropers":
            parts = line.split("#")
            data_part = parts[0].split()
            comment = parts[1].strip() if len(parts) > 1 else ""
            if len(data_part) >= 6:
                sections["impropers"].append((
                    int(data_part[1]), int(data_part[2]),
                    int(data_part[3]), int(data_part[4]),
                    int(data_part[5]), comment))

        i += 1

    # Also extract counts from header
    for line in sections["header"]:
        line = line.strip()
        if "atoms" in line:
            parts = line.split()
            sections["counts"] = int(parts[0])
            break

    return sections


def get_charmm_type(comment):
    """Extract CHARMM type name from an atom comment string."""
    parts = comment.split("-")
    if len(parts) > 0:
        last = parts[-1]
        return last
    return None


def get_molecule_label(comment):
    """Extract molecule label prefix from comment (e.g., 'HAB', 'SV1', 'HAA', 'SOLV')."""
    parts = comment.split("-")
    if len(parts) > 0:
        return parts[0]
    return None


def get_molecule_id(atom_lines):
    """Determine molecule IDs by grouping atoms that share the same mol_id field."""
    mol_map = OrderedDict()
    for atom in atom_lines:
        mol_id = atom[1]
        if mol_id not in mol_map:
            mol_map[mol_id] = []
        mol_map[mol_id].append(atom)
    return mol_map


def extract_molecule_template(data, mol_label, species_name, max_molecules=None):
    """
    Extract one molecule template (first occurrence) of the given species.

    Returns: (atoms, bonds, angles, dihedrals, impropers, num_atoms)
    where atoms/bonds/angles/dihedrals are lists relative to this molecule.
    """
    # Group atoms by molecule ID
    mol_atoms = {}
    for atom in data["atoms"]:
        label = get_molecule_label(atom[7])
        if label == mol_label:
            mol_id = atom[1]
            if mol_id not in mol_atoms:
                mol_atoms[mol_id] = []
            mol_atoms[mol_id].append(atom)

    if not mol_atoms:
        return [], [], [], [], [], 0

    molecules = list(mol_atoms.items())
    templates = []

    for mol_idx, (mol_id, atoms) in enumerate(molecules[:max_molecules or 1]):
        atom_id_set = set(a[0] for a in atoms)
        natoms = len(atoms)

        mol_bonds = [b for b in data["bonds"] if b[1] in atom_id_set and b[2] in atom_id_set]
        mol_angles = [a for a in data["angles"] if a[1] in atom_id_set and a[2] in atom_id_set and a[3] in atom_id_set]
        mol_dihedrals = [d for d in data["dihedrals"] if all(x in atom_id_set for x in d[1:5])]
        mol_impropers = [imp for imp in data["impropers"] if all(x in atom_id_set for x in imp[1:5])]

        templates.append((atoms, mol_bonds, mol_angles, mol_dihedrals, mol_impropers, natoms))

    return templates


def random_rotation():
    """Generate a random 3D rotation matrix (pure Python, no numpy dependency)."""
    theta = random.uniform(0, 2 * math.pi)
    phi = random.uniform(0, math.pi)
    psi = random.uniform(0, 2 * math.pi)

    cx, sx = math.cos(theta), math.sin(theta)
    cy, sy = math.cos(phi), math.sin(phi)
    cz, sz = math.cos(psi), math.sin(psi)

    # R = Rz @ Ry @ Rx
    return [
        [cy*cz, cz*sx*sy - cx*sz, cx*cz*sy + sx*sz],
        [cy*sz, cx*cz + sx*sy*sz, -cz*sx + cx*sy*sz],
        [-sy, cy*sx, cx*cy],
    ]

def matvec(M, v):
    """Multiply 3x3 matrix by 3-vector."""
    return [M[0][0]*v[0] + M[0][1]*v[1] + M[0][2]*v[2],
            M[1][0]*v[0] + M[1][1]*v[1] + M[1][2]*v[2],
            M[2][0]*v[0] + M[2][1]*v[1] + M[2][2]*v[2]]


def random_placement_3d(box_edge, margin=2.0, max_attempts=1000):
    """Generate random non-overlapping positions within a box."""
    return (random.uniform(-box_edge/2 + margin, box_edge/2 - margin),
            random.uniform(-box_edge/2 + margin, box_edge/2 - margin),
            random.uniform(-box_edge/2 + margin, box_edge/2 - margin))


def build_unified_type_map(source_data, source_name):
    """
    Build a mapping from source atom type -> unified type
    by parsing CHARMM type names from atom comments.
    """
    type_map = {}
    seen_types = set()
    for atom in source_data["atoms"]:
        src_type = atom[2]
        if src_type in seen_types:
            continue
        charmm_type = get_charmm_type(atom[7])
        if charmm_type in CHARMM_TYPE_TO_UNIFIED:
            type_map[src_type] = CHARMM_TYPE_TO_UNIFIED[charmm_type]
            seen_types.add(src_type)
        else:
            print(f"WARNING: unknown CHARMM type '{charmm_type}' from {source_name} atom {atom[0]}")
    return type_map


def build_pair_coeffs_unified(etoh_data, water_data):
    """
    Build unified pair coefficients from source files.

    CHARMM LAMMPS data commonly stores epsilon/sigma plus 1-4 epsilon/sigma.
    Keep all provided values instead of duplicating the normal LJ parameters.
    """
    coeffs = {}
    for source_name, data in [("EtOH", etoh_data), ("Water", water_data)]:
        if data["pair_coeffs"] is None:
            continue

        source_type_names = {}
        for atom in data["atoms"]:
            source_type_names.setdefault(atom[2], get_charmm_type(atom[7]))

        for src_type, params in sorted(data["pair_coeffs"].items()):
            charmm_type = source_type_names.get(src_type)
            if charmm_type not in CHARMM_TYPE_TO_UNIFIED:
                print(f"WARNING: unknown CHARMM pair type '{charmm_type}' from {source_name}")
                continue

            unified_type = CHARMM_TYPE_TO_UNIFIED[charmm_type]
            old = coeffs.get(unified_type)
            if old is not None:
                old_key = tuple(round(x, 12) for x in old)
                new_key = tuple(round(x, 12) for x in params)
                if old_key != new_key:
                    print(
                        f"WARNING: conflicting pair coeffs for {charmm_type}: "
                        f"{old} vs {params}; keeping {old}")
                    continue
            coeffs[unified_type] = params

    for unified_type in range(1, 14):
        if unified_type not in coeffs:
            eps, sig = PAIR_COEFFS[unified_type]
            coeffs[unified_type] = (eps, sig, eps, sig)

    return OrderedDict((t, coeffs[t]) for t in range(1, 14))


def interleave_molecules(template_atom_lists, box_edge, min_atom_distance=1.2):
    """
    Place molecules at non-overlapping random positions in the box.

    A small cell list catches hard inter-molecular clashes in the initial
    geometry. Molecules are still centered and rotated in build_merged_config().

    Returns: list of (translated atoms, rotation_matrix, center)
    """
    total = len(template_atom_lists)
    if total == 0:
        return []

    def centered_template_coords(template):
        atoms = template[0]
        cx = sum(atom[4] for atom in atoms) / len(atoms)
        cy = sum(atom[5] for atom in atoms) / len(atoms)
        cz = sum(atom[6] for atom in atoms) / len(atoms)
        coords = [(atom[4] - cx, atom[5] - cy, atom[6] - cz)
                  for atom in atoms]
        radius = max(math.sqrt(x*x + y*y + z*z) for x, y, z in coords)
        return coords, radius

    prepared = []
    for _, _, template in template_atom_lists:
        prepared.append(centered_template_coords(template))

    half = box_edge / 2.0
    min_dist2 = min_atom_distance * min_atom_distance
    cell = min_atom_distance
    grid = {}

    def cell_key(x, y, z):
        return (math.floor(x / cell), math.floor(y / cell), math.floor(z / cell))

    def has_clash(coords):
        for x, y, z in coords:
            ix, iy, iz = cell_key(x, y, z)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        for px, py, pz in grid.get((ix + dx, iy + dy, iz + dz), []):
                            d2 = (x - px) ** 2 + (y - py) ** 2 + (z - pz) ** 2
                            if d2 < min_dist2:
                                return True
        return False

    def add_to_grid(coords):
        for x, y, z in coords:
            grid.setdefault(cell_key(x, y, z), []).append((x, y, z))

    results = []
    for idx, (rel_coords, radius) in enumerate(prepared):
        limit = half - radius - min_atom_distance
        if limit <= 0:
            raise ValueError(
                f"Box edge {box_edge} A is too small for molecule radius {radius:.3f} A")

        for attempt in range(10000):
            cx = random.uniform(-limit, limit)
            cy = random.uniform(-limit, limit)
            cz = random.uniform(-limit, limit)
            rot = random_rotation()
            coords = []
            for rel in rel_coords:
                rx, ry, rz = matvec(rot, rel)
                coords.append((rx + cx, ry + cy, rz + cz))

            if not has_clash(coords):
                add_to_grid(coords)
                results.append((cx, cy, cz, rot))
                break
        else:
            raise RuntimeError(
                f"Could not place molecule {idx + 1}/{total} without atom clashes; "
                f"try a larger --box or a smaller min_atom_distance.")

    return results


def build_bond_coeffs_unified(etoh_data, water_data):
    """
    Build unified bond coefficient table from both source files.
    Uses parameter matching to avoid duplicates.
    """
    from collections import OrderedDict

    # Key: (K, r0) -> unified bond type
    param_to_type = OrderedDict()
    type_counter = 1

    for src_name, data in [("etoh", etoh_data), ("water", water_data)]:
        if data["bond_coeffs"] is None:
            continue
        for src_type, params in sorted(data["bond_coeffs"].items()):
            key = (round(params[0], 4), round(params[1], 6))
            if key not in param_to_type:
                param_to_type[key] = type_counter
                type_counter += 1

    # Build mapping: (src_name, src_type) -> unified_type
    mapping = {}
    for src_name, data in [("etoh", etoh_data), ("water", water_data)]:
        if data["bond_coeffs"] is None:
            continue
        for src_type, params in sorted(data["bond_coeffs"].items()):
            key = (round(params[0], 4), round(params[1], 6))
            mapping[(src_name, src_type)] = param_to_type[key]

    # Build type_to_params dict {unified_type: (K, r0)}
    type_to_params = OrderedDict()
    for params, utype in param_to_type.items():
        type_to_params[utype] = params

    return type_to_params, mapping


def build_angle_coeffs_unified(etoh_data, water_data):
    """Build unified angle coefficient table."""
    from collections import OrderedDict
    param_to_type = OrderedDict()
    type_counter = 1
    for src_name, data in [("etoh", etoh_data), ("water", water_data)]:
        if data["angle_coeffs"] is None:
            continue
        for src_type, params in sorted(data["angle_coeffs"].items()):
            key = tuple(round(p, 5) for p in params)
            if key not in param_to_type:
                param_to_type[key] = type_counter
                type_counter += 1
    mapping = {}
    for src_name, data in [("etoh", etoh_data), ("water", water_data)]:
        if data["angle_coeffs"] is None:
            continue
        for src_type, params in sorted(data["angle_coeffs"].items()):
            key = tuple(round(p, 5) for p in params)
            mapping[(src_name, src_type)] = param_to_type[key]
    type_to_params = OrderedDict()
    for params, utype in param_to_type.items():
        type_to_params[utype] = params
    return type_to_params, mapping


def build_dihedral_coeffs_unified(etoh_data, water_data):
    """Build unified dihedral coefficient table."""
    from collections import OrderedDict
    param_to_type = OrderedDict()
    type_counter = 1
    for src_name, data in [("etoh", etoh_data), ("water", water_data)]:
        if data["dihedral_coeffs"] is None:
            continue
        for src_type, params in sorted(data["dihedral_coeffs"].items()):
            key = tuple(round(p, 5) for p in params)
            if key not in param_to_type:
                param_to_type[key] = type_counter
                type_counter += 1
    mapping = {}
    for src_name, data in [("etoh", etoh_data), ("water", water_data)]:
        if data["dihedral_coeffs"] is None:
            continue
        for src_type, params in sorted(data["dihedral_coeffs"].items()):
            key = tuple(round(p, 5) for p in params)
            mapping[(src_name, src_type)] = param_to_type[key]
    type_to_params = OrderedDict()
    for params, utype in param_to_type.items():
        type_to_params[utype] = params
    return type_to_params, mapping


def resolve_water_shake_types(water_template, bond_map, angle_map):
    """
    Return (oh_bond_type, hoh_angle_type): the unified type numbers assigned to
    the TIP3P water O-H bond and H-O-H angle.

    Identifies them by CHARMM atom-type names in the template comments (OT/HT),
    then maps through the same bond/angle dedup mappings used for the topology,
    so the numbers always match what the run input must SHAKE.
    """
    _, w_bonds, w_angles, _, _, _ = water_template

    oh_bond_types = set()
    for b in w_bonds:
        names = [get_charmm_type(tok) for tok in b[3].split()]
        if "OT" in names and "HT" in names:
            oh_bond_types.add(bond_map.get(("water", b[0]), b[0]))

    hoh_angle_types = set()
    for a in w_angles:
        names = [get_charmm_type(tok) for tok in a[4].split()]
        if names.count("HT") >= 2 and "OT" in names:
            hoh_angle_types.add(angle_map.get(("water", a[0]), a[0]))

    if len(oh_bond_types) != 1:
        raise ValueError(
            f"Expected exactly one water O-H bond type, found {sorted(oh_bond_types)}")
    if len(hoh_angle_types) != 1:
        raise ValueError(
            f"Expected exactly one water H-O-H angle type, found {sorted(hoh_angle_types)}")

    return oh_bond_types.pop(), hoh_angle_types.pop()


def build_merged_config(etoh_file, water_file, n_para, n_water, n_etoh,
                        box_edge, seed, output_file):
    """
    Build the merged input-config.dat from template molecules.

    Paracetamol template comes from the EtOH file (HAB1 label).
    Ethanol template comes from the EtOH file (SV1 label).
    Water template comes from the Water file (SOLV label).
    """
    import random as _random
    _random.seed(seed)

    print(f"Reading EtOH config: {etoh_file}")
    etoh_data = parse_data_file(etoh_file)
    print(f"Reading Water config: {water_file}")
    water_data = parse_data_file(water_file)

    # --- Extract molecule templates ---
    para_templates = extract_molecule_template(etoh_data, "HAB1", "paracetamol", max_molecules=1)
    etoh_templates = extract_molecule_template(etoh_data, "SV1", "ethanol", max_molecules=1)
    water_templates = extract_molecule_template(water_data, "SOLV", "water", max_molecules=1)

    if not para_templates:
        print("ERROR: No paracetamol (HAB1) found in EtOH file")
        sys.exit(1)
    if not etoh_templates:
        print("ERROR: No ethanol (SV1) found in EtOH file")
        sys.exit(1)
    if not water_templates:
        print("ERROR: No water (SOLV) found in Water file")
        sys.exit(1)

    para_template = para_templates[0]   # (atoms, bonds, angles, dihedrals, impropers, natoms)
    etoh_template = etoh_templates[0]
    water_template = water_templates[0]

    # --- Build type maps ---
    etoh_type_map = build_unified_type_map(etoh_data, "EtOH")
    water_type_map = build_unified_type_map(water_data, "Water")

    # --- Build unified coeffs tables and mappings ---
    pair_coeffs = build_pair_coeffs_unified(etoh_data, water_data)
    bond_coeffs, bond_map = build_bond_coeffs_unified(etoh_data, water_data)
    angle_coeffs, angle_map = build_angle_coeffs_unified(etoh_data, water_data)
    dihedral_coeffs, dihedral_map = build_dihedral_coeffs_unified(etoh_data, water_data)

    # --- Resolve the unified type numbers the run input must SHAKE ---
    # input-script.dat hard-codes `fix ... shake ... b <bond> a <angle>` for the
    # rigid water. Those indices depend on the dedup outcome, so detect them from
    # the water template instead of trusting a literal that silently goes stale.
    shake_bond_type, shake_angle_type = resolve_water_shake_types(
        water_template, bond_map, angle_map)

    # --- Build molecule lists with coordinates ---
    mol_atom_groups = []  # Each: (source_type_map, source_name, template, etoh/water)

    for _ in range(n_para):
        mol_atom_groups.append((etoh_type_map, "etoh", para_template))
    for _ in range(n_etoh):
        mol_atom_groups.append((etoh_type_map, "etoh", etoh_template))
    for _ in range(n_water):
        mol_atom_groups.append((water_type_map, "water", water_template))

    total_molecules = len(mol_atom_groups)
    placements = interleave_molecules(mol_atom_groups, box_edge)

    # --- Assemble ---
    new_atoms = []
    new_bonds = []
    new_angles = []
    new_dihedrals = []
    new_impropers = []

    atom_offset = 0
    mol_offset = 0

    for idx, (type_map, src_name, template) in enumerate(mol_atom_groups):
        template_atoms, template_bonds, template_angles, template_dihedrals, \
            template_impropers, _ = template

        cx, cy, cz, rot = placements[idx]

        # Rotate about the template molecule center, not the source box origin.
        center_x = sum(atom[4] for atom in template_atoms) / len(template_atoms)
        center_y = sum(atom[5] for atom in template_atoms) / len(template_atoms)
        center_z = sum(atom[6] for atom in template_atoms) / len(template_atoms)

        # Build old_id -> new_id map for this molecule
        old_to_new = {}
        for atom in template_atoms:
            old_id = atom[0]
            new_id = atom_offset + len(old_to_new) + 1
            old_to_new[old_id] = new_id

            src_type = atom[2]
            unified_type = type_map.get(src_type, src_type)
            if unified_type not in MASSES:
                print(f"WARNING: no mass for type {unified_type}")

            # Apply rotation and translation
            x = atom[4] - center_x
            y = atom[5] - center_y
            z = atom[6] - center_z
            nx, ny, nz = matvec(rot, [x, y, z])
            nx += cx
            ny += cy
            nz += cz

            # Build new comment with updated molecule label
            new_mol_id = idx + 1
            comment = atom[7]

            new_atoms.append((
                new_id, new_mol_id, unified_type, atom[3],
                nx, ny, nz, comment))

        atom_offset += len(template_atoms)

        # Remap bonded interactions
        for b in template_bonds:
            if b[1] in old_to_new and b[2] in old_to_new:
                src_btype = b[0]
                unified_btype = bond_map.get((src_name, src_btype), src_btype)
                new_bonds.append((
                    unified_btype, old_to_new[b[1]], old_to_new[b[2]], b[3]))

        for a in template_angles:
            if all(x in old_to_new for x in [a[1], a[2], a[3]]):
                src_atype = a[0]
                unified_atype = angle_map.get((src_name, src_atype), src_atype)
                new_angles.append((
                    unified_atype, old_to_new[a[1]], old_to_new[a[2]],
                    old_to_new[a[3]], a[4]))

        for d in template_dihedrals:
            if all(x in old_to_new for x in d[1:5]):
                src_dtype = d[0]
                unified_dtype = dihedral_map.get((src_name, src_dtype), src_dtype)
                new_dihedrals.append((
                    unified_dtype, old_to_new[d[1]], old_to_new[d[2]],
                    old_to_new[d[3]], old_to_new[d[4]], d[5]))

        for imp in template_impropers:
            if all(x in old_to_new for x in imp[1:5]):
                new_impropers.append((
                    imp[0], old_to_new[imp[1]], old_to_new[imp[2]],
                    old_to_new[imp[3]], old_to_new[imp[4]], imp[5]))

    # --- Calculate counts ---
    total_atoms = len(new_atoms)
    total_bonds = len(new_bonds)
    total_angles = len(new_angles)
    total_dihedrals = len(new_dihedrals)
    total_impropers = len(new_impropers)

    n_atom_types = 13
    n_bond_types = len(bond_coeffs)
    n_angle_types = len(angle_coeffs)
    n_dihedral_types = len(dihedral_coeffs)

    # This script has no Improper Coeffs / unification path. The supported
    # templates carry no impropers, so 0 is correct. If a future template does,
    # fail loudly rather than emit an Impropers section with no declared types.
    if new_impropers:
        print("ERROR: templates contain impropers, but this script has no "
              "Improper Coeffs handling. Refusing to write an inconsistent file "
              "(Impropers section with 0 improper types).")
        sys.exit(1)
    n_improper_types = 0

    # --- Write output ---
    half = box_edge / 2.0
    with open(output_file, "w") as f:
        # LAMMPS ignores the first line (title); use it to record the SHAKE types
        # so `fix shake ... b <bond> a <angle>` in the run input can be verified.
        f.write(f"# SHAKE: water O-H bond type = {shake_bond_type}, "
                f"H-O-H angle type = {shake_angle_type}\n")
        f.write(f"    {total_atoms:>12}  atoms\n")
        f.write(f"    {total_bonds:>12}  bonds\n")
        f.write(f"    {total_angles:>12}  angles\n")
        f.write(f"    {total_dihedrals:>12}  dihedrals\n")
        f.write(f"    {total_impropers:>12}  impropers\n")
        f.write(f"                 \n")
        f.write(f"                 \n")
        f.write(f"    {n_atom_types:>12}  atom types\n")
        f.write(f"    {n_bond_types:>12}  bond types\n")
        f.write(f"    {n_angle_types:>12}  angle types\n")
        f.write(f"    {n_dihedral_types:>12}  dihedral types\n")
        f.write(f"    {n_improper_types:>12}  improper types\n")
        f.write(f"\n")
        f.write(f"  {-half:12.4f}    {half:12.4f} xlo xhi\n")
        f.write(f"  {-half:12.4f}    {half:12.4f} ylo yhi\n")
        f.write(f"  {-half:12.4f}    {half:12.4f} zlo zhi\n")
        f.write(f"\n")
        f.write(f"Masses\n")
        f.write(f"\n")
        for t in range(1, 14):
            f.write(f"       {t}       {MASSES[t]:.3f}  # {UNIFIED_TO_CHARMM_TYPE[t]}\n")
        f.write(f"\n")
        f.write(f"Pair Coeffs\n")
        f.write(f"\n")
        for t in range(1, 14):
            params = pair_coeffs[t]
            parts = " ".join(f"{p:.14f}" for p in params)
            f.write(f"       {t} {parts} # {UNIFIED_TO_CHARMM_TYPE[t]}\n")
        f.write(f"\n")
        f.write(f"Atoms\n")
        f.write(f"\n")
        for atom in new_atoms:
            f.write(f"  {atom[0]:>10} {atom[1]:>10} {atom[2]:>6} {atom[3]:.3f}"
                    f"   {atom[4]:>18.10f} {atom[5]:>18.10f} {atom[6]:>18.10f}"
                    f" # {atom[7]}\n")

        if bond_coeffs:
            f.write(f"\nBond Coeffs\n\n")
            for t, (k, r0) in sorted(bond_coeffs.items()):
                f.write(f"  {t:>10} {k:>10.3f}  {r0:.4f}\n")

        if new_bonds:
            f.write(f"\nBonds\n\n")
            for bid, b in enumerate(new_bonds, 1):
                f.write(f"  {bid:>10} {b[0]:>6} {b[1]:>10} {b[2]:>10} # {b[3]}\n")

        if angle_coeffs:
            f.write(f"\nAngle Coeffs\n\n")
            for t, params in sorted(angle_coeffs.items()):
                parts = " ".join(f"{p:.5f}" if isinstance(p, float) else str(p) for p in params)
                f.write(f"  {t:>10} {parts}\n")

        if new_angles:
            f.write(f"\nAngles\n\n")
            for aid, a in enumerate(new_angles, 1):
                f.write(f"  {aid:>12} {a[0]:>6} {a[1]:>12} {a[2]:>12} {a[3]:>12} # {a[4]}\n")

        if dihedral_coeffs:
            f.write(f"\nDihedral Coeffs\n\n")
            for t, params in sorted(dihedral_coeffs.items()):
                # dihedral_style charmm: K(real) n(int) d(int degrees) weight(real)
                K, n, d, w = params
                f.write(f"  {t:>10} {K:.5f} {int(round(n))} {int(round(d))} {w:.5f}\n")

        if new_dihedrals:
            f.write(f"\nDihedrals\n\n")
            for did, d in enumerate(new_dihedrals, 1):
                f.write(f"  {did:>12} {d[0]:>6} {d[1]:>12} {d[2]:>12} {d[3]:>12} {d[4]:>12} # {d[5]}\n")

        if new_impropers:
            f.write(f"\nImpropers\n\n")
            for iid, imp in enumerate(new_impropers, 1):
                f.write(f"  {iid:>12} {imp[0]:>6} {imp[1]:>12} {imp[2]:>12} {imp[3]:>12} {imp[4]:>12} # {imp[5]}\n")

    print(f"\nWrote {output_file}:")
    print(f"  {total_atoms} atoms ({n_para} para + {n_water} water + {n_etoh} EtOH)")
    print(f"  {total_bonds} bonds, {total_angles} angles, {total_dihedrals} dihedrals")
    print(f"  Box: [{-half}, {half}]")
    print(f"  Atom types: {n_atom_types} (unified CHARMM36)")
    print(f"  Solvent molecules: {n_water + n_etoh}")
    print(f"  SHAKE types for input-script.dat: water O-H bond = {shake_bond_type}, "
          f"H-O-H angle = {shake_angle_type}  (must match `fix ... shake ... b <..> a <..>`)")

MOLECULE_TEMPLATE = '# Compact validated templates: one HAB1, one SV1, one SOLV molecule\n32 atoms\n30 bonds\n45 angles\n56 dihedrals\n0 impropers\n\n13 atom types\n14 bond types\n18 angle types\n16 dihedral types\n0 improper types\n\n-30.0 30.0 xlo xhi\n-30.0 30.0 ylo yhi\n-30.0 30.0 zlo zhi\n\nMasses\n\n1 1.00800000\n2 1.00800000\n3 1.00800000\n4 1.00800000\n5 1.00800000\n6 12.01100000\n7 12.01100000\n8 12.01100000\n9 12.01100000\n10 14.00700000\n11 15.99900000\n12 15.99900000\n13 15.99900000\n\nPair Coeffs\n\n1 0.04600000000000 0.40001352444501 0.04600000000000 0.40001352444501\n2 0.03400000000000 2.38760856461611 0.03400000000000 2.38760856461611\n3 0.02400000000000 2.38760856461611 0.02400000000000 2.38760856461611\n4 0.03000000000000 2.42003727795642 0.03000000000000 2.42003727795642\n5 0.04600000000000 0.40001352444501 0.04600000000000 0.40001352444501\n6 0.11000000000000 3.56359487256136 0.11000000000000 3.56359487256136\n7 0.07000000000000 3.55005321204562 0.07000000000000 3.55005321204562\n8 0.05600000000000 3.58141284692416 0.01000000000000 3.38541512893329\n9 0.07800000000000 3.63486677001258 0.01000000000000 3.38541512893329\n10 0.20000000000000 3.29632525711926 0.20000000000000 2.76178602623505\n11 0.12000000000000 3.02905564167715 0.12000000000000 2.49451641079295\n12 0.15210000000000 3.15378146221680 0.15210000000000 3.15378146221680\n13 0.15210000000000 3.15057422683150 0.15210000000000 3.15057422683150\n\nAtoms\n\n1 1 9 -0.270000000000 -22.392010601300 -5.920712913900 13.556208463200 # HAB1-1-PACP-C14-CT3\n2 1 3 0.090000000000 -23.088075164700 -5.213480219100 13.052292597500 # HAB1-1-PACP-H141-HA3\n3 1 3 0.090000000000 -21.369072076700 -5.492981630000 13.524560523200 # HAB1-1-PACP-H142-HA3\n4 1 3 0.090000000000 -22.713653316500 -6.045238708600 14.610333774500 # HAB1-1-PACP-H143-HA3\n5 1 6 0.520000000000 -22.443760983700 -7.222931280900 12.840059885000 # HAB1-1-PACP-C15-C\n6 1 11 -0.520000000000 -23.236979273300 -7.377492337900 11.931286988200 # HAB1-1-PACP-O29-O\n7 1 10 -0.470000000000 -21.574451395300 -8.173075912200 13.246102387800 # HAB1-1-PACP-N21-NH1\n8 1 1 0.330000000000 -21.003253531600 -7.993038687900 14.029174348600 # HAB1-1-PACP-H211-H\n9 1 7 0.140000000000 -21.382123938300 -9.445525584300 12.632522638100 # HAB1-1-PACP-C22-CA\n10 1 7 -0.115000000000 -21.153740413200 -10.552822572400 13.464794845300 # HAB1-1-PACP-C23-CA\n11 1 4 0.115000000000 -21.096406148600 -10.416988294900 14.532746095600 # HAB1-1-PACP-H231-HP\n12 1 7 -0.115000000000 -21.060085155000 -11.843318998600 12.914274926800 # HAB1-1-PACP-C24-CA\n13 1 4 0.115000000000 -20.957519530300 -12.713826889500 13.538346329000 # HAB1-1-PACP-H241-HP\n14 1 7 -0.115000000000 -21.370963523700 -10.935416730500 10.695403942100 # HAB1-1-PACP-C26-CA\n15 1 4 0.115000000000 -21.462236737400 -11.067854048600 9.627394108900 # HAB1-1-PACP-H261-HP\n16 1 7 -0.115000000000 -21.487115479800 -9.649335688600 11.247813992100 # HAB1-1-PACP-C27-CA\n17 1 4 0.115000000000 -21.667690980400 -8.805487706200 10.600216676600 # HAB1-1-PACP-H271-HP\n18 1 7 0.110000000000 -21.160608526200 -12.034272702500 11.536577883600 # HAB1-1-PACP-C25-CA\n19 1 12 -0.540000000000 -21.075384393800 -13.343605167300 11.018788957000 # HAB1-1-PACP-O28-OH1\n20 1 1 0.430000000000 -21.194180204700 -13.276331477400 10.069015166600 # HAB1-1-PACP-H281-H\n521 27 8 0.050000000000 -24.637363166400 14.819505068200 -12.673795337700 # SV1-1-ETOH-C1-CT2\n522 27 12 -0.660000000000 -25.971039899500 14.992658160500 -12.202995287000 # SV1-1-ETOH-O1-OH1\n523 27 1 0.430000000000 -26.039388918200 15.886676949600 -11.820320109300 # SV1-1-ETOH-HO1-H\n524 27 2 0.090000000000 -23.919834042800 14.869183966200 -11.820921969900 # SV1-1-ETOH-H11-HA2\n525 27 2 0.090000000000 -24.386788361500 15.619050574500 -13.404181762700 # SV1-1-ETOH-H12-HA2\n526 27 9 -0.270000000000 -24.508188278700 13.471886756600 -13.376463144100 # SV1-1-ETOH-C2-CT3\n527 27 3 0.090000000000 -24.940929844400 13.517528503200 -14.397814256100 # SV1-1-ETOH-H21-HA3\n528 27 3 0.090000000000 -25.032290354000 12.667600723500 -12.816465256200 # SV1-1-ETOH-H22-HA3\n529 27 3 0.090000000000 -23.436720512600 13.194169600600 -13.456321395800 # SV1-1-ETOH-H23-HA3\n1646 152 13 -0.834000000000 21.504238514800 -25.920473302200 -1.489190700000 # SOLV-1-TIP3-OH2-OT\n1647 152 5 0.417000000000 21.954788128500 -25.056446992900 -1.606021538100 # SOLV-1-TIP3-H1-HT\n1648 152 5 0.417000000000 20.826567962400 -25.707810121700 -0.831050479300 # SOLV-1-TIP3-H2-HT\n\nBond Coeffs\n\n1 305.000000000000 1.375000000000\n2 305.000000000000 1.414000000000\n3 250.000000000000 1.490000000000\n4 222.500000000000 1.528000000000\n5 309.000000000000 1.111000000000\n6 322.000000000000 1.111000000000\n7 340.000000000000 1.080000000000\n8 370.000000000000 1.345000000000\n9 440.000000000000 0.997000000000\n10 620.000000000000 1.230000000000\n11 334.300000000000 1.411000000000\n12 428.000000000000 1.420000000000\n13 545.000000000000 0.960000000000\n14 450.000000000000 0.957200000000\n\nBonds\n\n1 6 1 2 # HAB1-1-PACP-C14-CT3 HAB1-1-PACP-H141-HA3\n2 6 1 3 # HAB1-1-PACP-C14-CT3 HAB1-1-PACP-H142-HA3\n3 6 1 4 # HAB1-1-PACP-C14-CT3 HAB1-1-PACP-H143-HA3\n4 3 1 5 # HAB1-1-PACP-C14-CT3 HAB1-1-PACP-C15-C\n5 10 5 6 # HAB1-1-PACP-C15-C HAB1-1-PACP-O29-O\n6 8 5 7 # HAB1-1-PACP-C15-C HAB1-1-PACP-N21-NH1\n7 9 7 8 # HAB1-1-PACP-N21-NH1 HAB1-1-PACP-H211-H\n8 2 7 9 # HAB1-1-PACP-N21-NH1 HAB1-1-PACP-C22-CA\n9 1 9 10 # HAB1-1-PACP-C22-CA HAB1-1-PACP-C23-CA\n10 1 9 16 # HAB1-1-PACP-C22-CA HAB1-1-PACP-C27-CA\n11 7 10 11 # HAB1-1-PACP-C23-CA HAB1-1-PACP-H231-HP\n12 1 10 12 # HAB1-1-PACP-C23-CA HAB1-1-PACP-C24-CA\n13 7 12 13 # HAB1-1-PACP-C24-CA HAB1-1-PACP-H241-HP\n14 1 12 18 # HAB1-1-PACP-C24-CA HAB1-1-PACP-C25-CA\n15 7 14 15 # HAB1-1-PACP-C26-CA HAB1-1-PACP-H261-HP\n16 1 14 16 # HAB1-1-PACP-C26-CA HAB1-1-PACP-C27-CA\n17 1 14 18 # HAB1-1-PACP-C26-CA HAB1-1-PACP-C25-CA\n18 7 16 17 # HAB1-1-PACP-C27-CA HAB1-1-PACP-H271-HP\n19 11 18 19 # HAB1-1-PACP-C25-CA HAB1-1-PACP-O28-OH1\n20 13 19 20 # HAB1-1-PACP-O28-OH1 HAB1-1-PACP-H281-H\n21 12 521 522 # SV1-1-ETOH-C1-CT2 SV1-1-ETOH-O1-OH1\n22 5 521 524 # SV1-1-ETOH-C1-CT2 SV1-1-ETOH-H11-HA2\n23 5 521 525 # SV1-1-ETOH-C1-CT2 SV1-1-ETOH-H12-HA2\n24 4 521 526 # SV1-1-ETOH-C1-CT2 SV1-1-ETOH-C2-CT3\n25 13 522 523 # SV1-1-ETOH-O1-OH1 SV1-1-ETOH-HO1-H\n26 6 526 527 # SV1-1-ETOH-C2-CT3 SV1-1-ETOH-H21-HA3\n27 6 526 528 # SV1-1-ETOH-C2-CT3 SV1-1-ETOH-H22-HA3\n28 6 526 529 # SV1-1-ETOH-C2-CT3 SV1-1-ETOH-H23-HA3\n29 14 1646 1647 # SOLV-1-TIP3-OH2-OT SOLV-1-TIP3-H1-HT\n30 14 1646 1648 # SOLV-1-TIP3-OH2-OT SOLV-1-TIP3-H2-HT\n\nAngle Coeffs\n\n1 40.000000000000 120.000000000000 35.000000000000 2.416200000000\n2 50.000000000000 120.000000000000 0.000000000000 0.000000000000\n3 34.000000000000 123.000000000000 0.000000000000 0.000000000000\n4 34.000000000000 117.000000000000 0.000000000000 0.000000000000\n5 65.000000000000 108.000000000000 0.000000000000 0.000000000000\n6 57.500000000000 106.000000000000 0.000000000000 0.000000000000\n7 34.600000000000 110.100000000000 22.530000000000 2.179000000000\n8 35.500000000000 109.000000000000 5.400000000000 1.802000000000\n9 33.000000000000 109.500000000000 30.000000000000 2.163000000000\n10 35.500000000000 108.400000000000 5.400000000000 1.802000000000\n11 30.000000000000 120.000000000000 22.000000000000 2.152500000000\n12 80.000000000000 116.500000000000 0.000000000000 0.000000000000\n13 80.000000000000 121.000000000000 0.000000000000 0.000000000000\n14 80.000000000000 122.500000000000 0.000000000000 0.000000000000\n15 45.200000000000 120.000000000000 0.000000000000 0.000000000000\n16 75.700000000000 110.100000000000 0.000000000000 0.000000000000\n17 45.900000000000 108.890000000000 0.000000000000 0.000000000000\n18 55.000000000000 104.520000000000 0.000000000000 0.000000000000\n\nAngles\n\n1 10 2 1 3 # HAB1-1-PACP-H141-HA3 HAB1-1-PACP-C14-CT3 HAB1-1-PACP-H142-HA3\n2 10 2 1 4 # HAB1-1-PACP-H141-HA3 HAB1-1-PACP-C14-CT3 HAB1-1-PACP-H143-HA3\n3 9 2 1 5 # HAB1-1-PACP-H141-HA3 HAB1-1-PACP-C14-CT3 HAB1-1-PACP-C15-C\n4 10 3 1 4 # HAB1-1-PACP-H142-HA3 HAB1-1-PACP-C14-CT3 HAB1-1-PACP-H143-HA3\n5 9 3 1 5 # HAB1-1-PACP-H142-HA3 HAB1-1-PACP-C14-CT3 HAB1-1-PACP-C15-C\n6 9 4 1 5 # HAB1-1-PACP-H143-HA3 HAB1-1-PACP-C14-CT3 HAB1-1-PACP-C15-C\n7 13 1 5 6 # HAB1-1-PACP-C14-CT3 HAB1-1-PACP-C15-C HAB1-1-PACP-O29-O\n8 12 1 5 7 # HAB1-1-PACP-C14-CT3 HAB1-1-PACP-C15-C HAB1-1-PACP-N21-NH1\n9 14 6 5 7 # HAB1-1-PACP-O29-O HAB1-1-PACP-C15-C HAB1-1-PACP-N21-NH1\n10 3 5 7 8 # HAB1-1-PACP-C15-C HAB1-1-PACP-N21-NH1 HAB1-1-PACP-H211-H\n11 2 5 7 9 # HAB1-1-PACP-C15-C HAB1-1-PACP-N21-NH1 HAB1-1-PACP-C22-CA\n12 4 8 7 9 # HAB1-1-PACP-H211-H HAB1-1-PACP-N21-NH1 HAB1-1-PACP-C22-CA\n13 1 7 9 10 # HAB1-1-PACP-N21-NH1 HAB1-1-PACP-C22-CA HAB1-1-PACP-C23-CA\n14 1 7 9 16 # HAB1-1-PACP-N21-NH1 HAB1-1-PACP-C22-CA HAB1-1-PACP-C27-CA\n15 1 10 9 16 # HAB1-1-PACP-C23-CA HAB1-1-PACP-C22-CA HAB1-1-PACP-C27-CA\n16 11 9 10 11 # HAB1-1-PACP-C22-CA HAB1-1-PACP-C23-CA HAB1-1-PACP-H231-HP\n17 1 9 10 12 # HAB1-1-PACP-C22-CA HAB1-1-PACP-C23-CA HAB1-1-PACP-C24-CA\n18 11 11 10 12 # HAB1-1-PACP-H231-HP HAB1-1-PACP-C23-CA HAB1-1-PACP-C24-CA\n19 11 10 12 13 # HAB1-1-PACP-C23-CA HAB1-1-PACP-C24-CA HAB1-1-PACP-H241-HP\n20 1 10 12 18 # HAB1-1-PACP-C23-CA HAB1-1-PACP-C24-CA HAB1-1-PACP-C25-CA\n21 11 13 12 18 # HAB1-1-PACP-H241-HP HAB1-1-PACP-C24-CA HAB1-1-PACP-C25-CA\n22 11 15 14 16 # HAB1-1-PACP-H261-HP HAB1-1-PACP-C26-CA HAB1-1-PACP-C27-CA\n23 11 15 14 18 # HAB1-1-PACP-H261-HP HAB1-1-PACP-C26-CA HAB1-1-PACP-C25-CA\n24 1 16 14 18 # HAB1-1-PACP-C27-CA HAB1-1-PACP-C26-CA HAB1-1-PACP-C25-CA\n25 1 9 16 14 # HAB1-1-PACP-C22-CA HAB1-1-PACP-C27-CA HAB1-1-PACP-C26-CA\n26 11 9 16 17 # HAB1-1-PACP-C22-CA HAB1-1-PACP-C27-CA HAB1-1-PACP-H271-HP\n27 11 14 16 17 # HAB1-1-PACP-C26-CA HAB1-1-PACP-C27-CA HAB1-1-PACP-H271-HP\n28 1 12 18 14 # HAB1-1-PACP-C24-CA HAB1-1-PACP-C25-CA HAB1-1-PACP-C26-CA\n29 15 12 18 19 # HAB1-1-PACP-C24-CA HAB1-1-PACP-C25-CA HAB1-1-PACP-O28-OH1\n30 15 14 18 19 # HAB1-1-PACP-C26-CA HAB1-1-PACP-C25-CA HAB1-1-PACP-O28-OH1\n31 5 18 19 20 # HAB1-1-PACP-C25-CA HAB1-1-PACP-O28-OH1 HAB1-1-PACP-H281-H\n32 17 522 521 524 # SV1-1-ETOH-O1-OH1 SV1-1-ETOH-C1-CT2 SV1-1-ETOH-H11-HA2\n33 17 522 521 525 # SV1-1-ETOH-O1-OH1 SV1-1-ETOH-C1-CT2 SV1-1-ETOH-H12-HA2\n34 16 522 521 526 # SV1-1-ETOH-O1-OH1 SV1-1-ETOH-C1-CT2 SV1-1-ETOH-C2-CT3\n35 8 524 521 525 # SV1-1-ETOH-H11-HA2 SV1-1-ETOH-C1-CT2 SV1-1-ETOH-H12-HA2\n36 7 524 521 526 # SV1-1-ETOH-H11-HA2 SV1-1-ETOH-C1-CT2 SV1-1-ETOH-C2-CT3\n37 7 525 521 526 # SV1-1-ETOH-H12-HA2 SV1-1-ETOH-C1-CT2 SV1-1-ETOH-C2-CT3\n38 6 521 522 523 # SV1-1-ETOH-C1-CT2 SV1-1-ETOH-O1-OH1 SV1-1-ETOH-HO1-H\n39 7 521 526 527 # SV1-1-ETOH-C1-CT2 SV1-1-ETOH-C2-CT3 SV1-1-ETOH-H21-HA3\n40 7 521 526 528 # SV1-1-ETOH-C1-CT2 SV1-1-ETOH-C2-CT3 SV1-1-ETOH-H22-HA3\n41 7 521 526 529 # SV1-1-ETOH-C1-CT2 SV1-1-ETOH-C2-CT3 SV1-1-ETOH-H23-HA3\n42 10 527 526 528 # SV1-1-ETOH-H21-HA3 SV1-1-ETOH-C2-CT3 SV1-1-ETOH-H22-HA3\n43 10 527 526 529 # SV1-1-ETOH-H21-HA3 SV1-1-ETOH-C2-CT3 SV1-1-ETOH-H23-HA3\n44 10 528 526 529 # SV1-1-ETOH-H22-HA3 SV1-1-ETOH-C2-CT3 SV1-1-ETOH-H23-HA3\n45 18 1647 1646 1648 # SOLV-1-TIP3-H1-HT SOLV-1-TIP3-OH2-OT SOLV-1-TIP3-H2-HT\n\nDihedral Coeffs\n\n1 1.200000000000 2.000000000000 180.000000000000 1.000000000000\n2 1.000000000000 3.000000000000 180.000000000000 0.000000000000\n3 3.100000000000 2.000000000000 180.000000000000 0.500000000000\n4 3.100000000000 2.000000000000 180.000000000000 1.000000000000\n5 2.500000000000 2.000000000000 180.000000000000 1.000000000000\n6 0.500000000000 2.000000000000 180.000000000000 1.000000000000\n7 0.990000000000 2.000000000000 180.000000000000 1.000000000000\n8 1.300000000000 1.000000000000 0.000000000000 1.000000000000\n9 0.300000000000 2.000000000000 0.000000000000 0.000000000000\n10 0.420000000000 3.000000000000 0.000000000000 0.000000000000\n11 0.160000000000 3.000000000000 0.000000000000 1.000000000000\n12 0.140000000000 3.000000000000 0.000000000000 1.000000000000\n13 4.200000000000 2.000000000000 180.000000000000 1.000000000000\n14 2.400000000000 2.000000000000 180.000000000000 1.000000000000\n15 0.000000000000 3.000000000000 0.000000000000 1.000000000000\n16 0.000000000000 3.000000000000 180.000000000000 1.000000000000\n\nDihedrals\n\n1 16 2 1 5 6 # H141-HA3 C14-CT3 C15-C O29-O\n2 15 2 1 5 7 # H141-HA3 C14-CT3 C15-C N21-NH1\n3 16 3 1 5 6 # H142-HA3 C14-CT3 C15-C O29-O\n4 15 3 1 5 7 # H142-HA3 C14-CT3 C15-C N21-NH1\n5 16 4 1 5 6 # H143-HA3 C14-CT3 C15-C O29-O\n6 15 4 1 5 7 # H143-HA3 C14-CT3 C15-C N21-NH1\n7 5 1 5 7 8 # C14-CT3 C15-C N21-NH1 H211-H\n8 5 1 5 7 9 # C14-CT3 C15-C N21-NH1 C22-CA\n9 5 6 5 7 8 # O29-O C15-C N21-NH1 H211-H\n10 5 6 5 7 9 # O29-O C15-C N21-NH1 C22-CA\n11 1 5 7 9 10 # C15-C N21-NH1 C22-CA C23-CA\n12 2 5 7 9 10 # C15-C N21-NH1 C22-CA C23-CA\n13 1 5 7 9 16 # C15-C N21-NH1 C22-CA C27-CA\n14 2 5 7 9 16 # C15-C N21-NH1 C22-CA C27-CA\n15 6 8 7 9 10 # H211-H N21-NH1 C22-CA C23-CA\n16 6 8 7 9 16 # H211-H N21-NH1 C22-CA C27-CA\n17 13 7 9 10 11 # N21-NH1 C22-CA C23-CA H231-HP\n18 4 7 9 10 12 # N21-NH1 C22-CA C23-CA C24-CA\n19 4 7 9 16 14 # N21-NH1 C22-CA C27-CA C26-CA\n20 13 7 9 16 17 # N21-NH1 C22-CA C27-CA H271-HP\n21 3 10 9 16 14 # C23-CA C22-CA C27-CA C26-CA\n22 13 10 9 16 17 # C23-CA C22-CA C27-CA H271-HP\n23 13 11 10 9 16 # H231-HP C23-CA C22-CA C27-CA\n24 3 12 10 9 16 # C24-CA C23-CA C22-CA C27-CA\n25 13 9 10 12 13 # C22-CA C23-CA C24-CA H241-HP\n26 3 9 10 12 18 # C22-CA C23-CA C24-CA C25-CA\n27 14 11 10 12 13 # H231-HP C23-CA C24-CA H241-HP\n28 13 11 10 12 18 # H231-HP C23-CA C24-CA C25-CA\n29 3 10 12 18 14 # C23-CA C24-CA C25-CA C26-CA\n30 4 10 12 18 19 # C23-CA C24-CA C25-CA O28-OH1\n31 13 13 12 18 14 # H241-HP C24-CA C25-CA C26-CA\n32 13 13 12 18 19 # H241-HP C24-CA C25-CA O28-OH1\n33 14 15 14 16 17 # H261-HP C26-CA C27-CA H271-HP\n34 13 15 14 18 19 # H261-HP C26-CA C25-CA O28-OH1\n35 4 16 14 18 19 # C27-CA C26-CA C25-CA O28-OH1\n36 13 9 16 14 15 # C22-CA C27-CA C26-CA H261-HP\n37 3 9 16 14 18 # C22-CA C27-CA C26-CA C25-CA\n38 13 17 16 14 18 # H271-HP C27-CA C26-CA C25-CA\n39 13 12 18 14 15 # C24-CA C25-CA C26-CA H261-HP\n40 3 12 18 14 16 # C24-CA C25-CA C26-CA C27-CA\n41 7 12 18 19 20 # C24-CA C25-CA O28-OH1 H281-H\n42 7 14 18 19 20 # C26-CA C25-CA O28-OH1 H281-H\n43 11 522 521 526 527 # O1-OH1 C1-CT2 C2-CT3 H21-HA3\n44 11 522 521 526 528 # O1-OH1 C1-CT2 C2-CT3 H22-HA3\n45 11 522 521 526 529 # O1-OH1 C1-CT2 C2-CT3 H23-HA3\n46 11 524 521 526 527 # H11-HA2 C1-CT2 C2-CT3 H21-HA3\n47 11 524 521 526 528 # H11-HA2 C1-CT2 C2-CT3 H22-HA3\n48 11 524 521 526 529 # H11-HA2 C1-CT2 C2-CT3 H23-HA3\n49 11 525 521 526 527 # H12-HA2 C1-CT2 C2-CT3 H21-HA3\n50 11 525 521 526 528 # H12-HA2 C1-CT2 C2-CT3 H22-HA3\n51 11 525 521 526 529 # H12-HA2 C1-CT2 C2-CT3 H23-HA3\n52 12 523 522 521 524 # HO1-H O1-OH1 C1-CT2 H11-HA2\n53 12 523 522 521 525 # HO1-H O1-OH1 C1-CT2 H12-HA2\n54 8 523 522 521 526 # HO1-H O1-OH1 C1-CT2 C2-CT3\n55 9 523 522 521 526 # HO1-H O1-OH1 C1-CT2 C2-CT3\n56 10 523 522 521 526 # HO1-H O1-OH1 C1-CT2 C2-CT3\n'


import contextlib
import hashlib
import io
import json
import re
import tempfile
from pathlib import Path


DEFAULT_OUTPUT = Path("MD_inputs")
MANIFEST_SCHEMA_VERSION = 1
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
REQUIRED_FIELDS = {
    "n_para", "n_water", "n_ethanol", "box_edge_angstrom", "seed"
}


def validate_level(level, composition):
    if not isinstance(level, str) or not NAME_RE.fullmatch(level):
        raise ValueError("level name must use only letters, numbers, dot, dash, and underscore")
    if not isinstance(composition, dict):
        raise ValueError("%s: composition must be a dictionary" % level)
    missing = sorted(REQUIRED_FIELDS - set(composition))
    extra = sorted(set(composition) - REQUIRED_FIELDS)
    if missing:
        raise ValueError("missing field(s): " + ", ".join(missing))
    if extra:
        raise ValueError("unknown field(s): " + ", ".join(extra))
    for field in ("n_para", "n_water", "n_ethanol", "seed"):
        if isinstance(composition[field], bool) or not isinstance(composition[field], int):
            raise ValueError("%s: %s must be an integer" % (level, field))
    if composition["n_para"] < 1:
        raise ValueError("%s: n_para must be at least 1" % level)
    if composition["n_water"] < 0 or composition["n_ethanol"] < 0:
        raise ValueError("%s: solvent counts cannot be negative" % level)
    if composition["n_water"] + composition["n_ethanol"] < 1:
        raise ValueError("%s: at least one solvent molecule is required" % level)
    edge = composition["box_edge_angstrom"]
    if (
        isinstance(edge, bool)
        or not isinstance(edge, (int, float))
        or not math.isfinite(edge)
        or edge <= 0
    ):
        raise ValueError("%s: box_edge_angstrom must be a finite positive number" % level)
    return dict(composition)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_composition_manifest(path):
    """Load the versioned external composition contract used by the CLI."""
    manifest_path = Path(path)
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError("composition manifest does not exist: %s" % manifest_path) from error
    except json.JSONDecodeError as error:
        raise ValueError(
            "%s is not valid JSON (line %d, column %d)"
            % (manifest_path, error.lineno, error.colno)
        ) from error
    if not isinstance(document, dict):
        raise ValueError("composition manifest must be a JSON object")
    allowed = {"schema_version", "level_composition"}
    unknown = sorted(set(document) - allowed)
    if unknown:
        raise ValueError("composition manifest has unknown field(s): " + ", ".join(unknown))
    schema_version = document.get("schema_version")
    if isinstance(schema_version, bool) or schema_version != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            "composition manifest schema_version must be %d" % MANIFEST_SCHEMA_VERSION
        )
    if "level_composition" not in document:
        raise ValueError("composition manifest is missing level_composition")
    levels = document["level_composition"]
    if not isinstance(levels, dict) or not levels:
        raise ValueError("composition manifest level_composition must be a non-empty object")
    return levels, {
        "name": manifest_path.name,
        "sha256": sha256_file(manifest_path),
    }


def header_atom_count(path):
    pattern = re.compile(r"^\s*(\d+)\s+atoms\s*$")
    with Path(path).open(encoding="utf-8") as handle:
        for _ in range(20):
            match = pattern.match(handle.readline())
            if match:
                return int(match.group(1))
    return None


def validate_existing_case(case_dir, expected_info):
    """Refuse to reuse output unless metadata and content checksum both match."""
    case_dir = Path(case_dir)
    data_path = case_dir / "input-config.dat"
    info_path = case_dir / "case_info.json"
    if not info_path.is_file():
        raise RuntimeError(
            "%s: input-config.dat exists without case_info.json; remove or move the case "
            "directory before regenerating" % expected_info["level"]
        )
    try:
        existing_info = json.loads(info_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError(
            "%s: case_info.json is invalid JSON; remove or move the case directory"
            % expected_info["level"]
        ) from error
    if not isinstance(existing_info, dict):
        raise RuntimeError("%s: case_info.json must contain an object" % expected_info["level"])

    mismatched = [
        key for key, value in expected_info.items()
        if key not in existing_info or existing_info[key] != value
    ]
    if mismatched:
        raise RuntimeError(
            "%s: existing input metadata differs for %s; use a new --output-root or "
            "remove/move that case directory"
            % (expected_info["level"], ", ".join(mismatched))
        )
    recorded_checksum = existing_info.get("input_sha256")
    if not isinstance(recorded_checksum, str) or not recorded_checksum:
        raise RuntimeError(
            "%s: case_info.json has no input_sha256; remove/move this unverified legacy case"
            % expected_info["level"]
        )
    actual_checksum = sha256_file(data_path)
    if actual_checksum != recorded_checksum:
        raise RuntimeError(
            "%s: input-config.dat checksum differs from case_info.json; refusing stale or "
            "modified input" % expected_info["level"]
        )
    return actual_checksum


def validate_output_root(output_root, expected_levels, composition_manifest, generator_sha256):
    """Require an existing root manifest to describe exactly this rerun."""
    output_root = Path(output_root)
    manifest_path = output_root / "manifest.json"
    entries = list(output_root.iterdir())
    if not manifest_path.exists():
        if entries:
            raise RuntimeError(
                "%s contains files but no manifest.json; use a new --output-root "
                "or move the legacy contents" % output_root
            )
        return
    try:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError(
            "%s is invalid JSON; refusing to reuse this output root" % manifest_path
        ) from error
    if not isinstance(existing, dict):
        raise RuntimeError("%s must contain a JSON object" % manifest_path)

    recorded_cases = existing.get("cases")
    if not isinstance(recorded_cases, list) or not all(
        isinstance(case, dict) and isinstance(case.get("level"), str)
        for case in recorded_cases
    ):
        raise RuntimeError("%s has no valid cases list" % manifest_path)
    recorded_levels = {case["level"] for case in recorded_cases}
    expected_levels = set(expected_levels)
    actual_levels = {entry.name for entry in entries if entry.is_dir()}
    mismatches = []
    if existing.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        mismatches.append("schema_version")
    if existing.get("composition_manifest") != composition_manifest:
        mismatches.append("composition_manifest")
    if existing.get("generator_sha256") != generator_sha256:
        mismatches.append("generator_sha256")
    if recorded_levels != expected_levels:
        mismatches.append("case set")
    if actual_levels != recorded_levels:
        mismatches.append("case directories")
    if mismatches:
        raise RuntimeError(
            "%s differs for %s; use a new --output-root or move the existing "
            "root before generating" % (manifest_path, ", ".join(mismatches))
        )


def build_levels(levels, output_root=DEFAULT_OUTPUT, quiet=False, composition_manifest=None):
    if not isinstance(levels, dict):
        raise ValueError("level_composition must be a dictionary")
    if not levels:
        raise ValueError(
            "level_composition is empty; provide --manifest FILE using "
            "md_input_manifest.schema.json"
        )

    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    template_bytes = MOLECULE_TEMPLATE.encode("utf-8")
    template_checksum = hashlib.sha256(template_bytes).hexdigest()
    generator_checksum = sha256_file(Path(__file__))
    validate_output_root(
        output_root,
        levels,
        composition_manifest,
        generator_checksum,
    )
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "case_count": len(levels),
        "composition_manifest": composition_manifest,
        "generator_sha256": generator_checksum,
        "cases": [],
    }

    with tempfile.TemporaryDirectory(prefix="md-input-template-") as temp_dir:
        template_path = Path(temp_dir) / "molecule-templates.dat"
        template_path.write_bytes(template_bytes)
        # General-use example: iterate over every named composition level.
        for index, (level, raw_composition) in enumerate(levels.items(), 1):
            composition = validate_level(level, raw_composition)
            case_dir = output_root / level
            case_dir.mkdir(parents=True, exist_ok=True)
            data_path = case_dir / "input-config.dat"
            n_atoms = (
                composition["n_para"] * 20
                + composition["n_water"] * 3
                + composition["n_ethanol"] * 9
            )
            info = {
                "level": level,
                **composition,
                "n_atoms_expected": n_atoms,
                "template_sha256": template_checksum,
                "generator_sha256": generator_checksum,
            }

            if data_path.exists():
                info["input_sha256"] = validate_existing_case(case_dir, info)
                status = "skip"
            else:
                stream = contextlib.nullcontext()
                if quiet:
                    stream = contextlib.redirect_stdout(io.StringIO())
                with stream:
                    build_merged_config(
                        etoh_file=str(template_path),
                        water_file=str(template_path),
                        n_para=composition["n_para"],
                        n_water=composition["n_water"],
                        n_etoh=composition["n_ethanol"],
                        box_edge=float(composition["box_edge_angstrom"]),
                        seed=composition["seed"],
                        output_file=str(data_path),
                    )
                status = "built"

            if header_atom_count(data_path) != n_atoms:
                raise RuntimeError(level + ": wrong generated atom count")
            with data_path.open(encoding="utf-8") as handle:
                shake_header = handle.readline()
            if "bond type = 14" not in shake_header or "angle type = 18" not in shake_header:
                raise RuntimeError(level + ": unexpected SHAKE type mapping")
            if status == "built":
                info["input_sha256"] = sha256_file(data_path)
                write_json(case_dir / "case_info.json", info)
            manifest["cases"].append(info)
            print("[%d/%d] %s %s" % (index, len(levels), level, status))

    write_json(output_root / "manifest.json", manifest)
    print("done: %s" % output_root)
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        help="versioned JSON composition manifest (recommended)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="output directory (default: %(default)s)",
    )
    parser.add_argument("--quiet", action="store_true", help="suppress generator details")
    return parser.parse_args(argv)


def standalone_main(argv=None):
    try:
        args = parse_args(argv)
        levels = level_composition
        source = {"kind": "inline level_composition", "sha256": None}
        if args.manifest is not None:
            levels, source = load_composition_manifest(args.manifest)
            source["kind"] = "file"
        return build_levels(
            levels,
            output_root=args.output_root,
            quiet=args.quiet,
            composition_manifest=source,
        )
    except (OSError, ValueError, RuntimeError) as error:
        raise SystemExit("ERROR: %s" % error)


if __name__ == "__main__":
    raise SystemExit(standalone_main())
