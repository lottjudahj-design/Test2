"""Minimal PDB parsing and fetching (numpy only, no Biopython required)."""
from __future__ import annotations

import os
import re
import urllib.request
from dataclasses import dataclass, field

import numpy as np

# Bondi-style van der Waals radii (Å); unknown elements fall back to 1.8.
VDW_RADII = {
    "H": 1.10, "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80,
    "F": 1.47, "CL": 1.75, "BR": 1.85, "I": 1.98, "SE": 1.90,
    "NA": 2.27, "K": 2.75, "MG": 1.73, "CA": 2.31, "ZN": 1.39, "FE": 1.94,
    "MN": 1.97, "CU": 1.40, "CO": 1.92, "NI": 1.63,
}
METALS = {"NA", "K", "MG", "CA", "ZN", "FE", "MN", "CU", "CO", "NI", "CD", "HG"}
WATER_NAMES = {"HOH", "WAT", "H2O", "DOD", "TIP", "TIP3", "SOL"}
# Common crystallisation additives / buffer molecules that are rarely the
# "ligand of interest". They are still loaded, just not auto-selected.
COMMON_ADDITIVES = {
    "SO4", "PO4", "GOL", "EDO", "PEG", "ACT", "DMS", "CL", "BR", "IOD", "FMT",
    "MPD", "TRS", "EPE", "MES", "NO3", "PG4", "1PE", "PGE", "BME", "IMD",
}
STANDARD_RESIDUES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "SEC", "PYL", "MSE", "DA", "DC", "DG", "DT", "A", "C", "G", "U",
}

_PDB_ID = re.compile(r"^[0-9][A-Za-z0-9]{3}$")


@dataclass
class Structure:
    """Column-oriented atom table for one model of a PDB entry."""

    name: str
    record: np.ndarray      # "ATOM" / "HETATM"
    serial: np.ndarray      # int
    atom_name: np.ndarray   # str
    resname: np.ndarray     # str
    chain: np.ndarray       # str
    resseq: np.ndarray      # int
    icode: np.ndarray       # str
    coords: np.ndarray      # (N, 3) float
    element: np.ndarray     # str, upper case
    pdb_text: str = ""      # cleaned PDB text (first model, single altloc)
    title: str = ""
    _radii: np.ndarray | None = field(default=None, repr=False)

    def __len__(self) -> int:
        return len(self.serial)

    @property
    def radii(self) -> np.ndarray:
        if self._radii is None:
            self._radii = np.array([VDW_RADII.get(e, 1.8) for e in self.element])
        return self._radii

    @property
    def is_water(self) -> np.ndarray:
        return np.isin(self.resname, list(WATER_NAMES))

    @property
    def is_hydrogen(self) -> np.ndarray:
        return np.isin(self.element, ["H", "D"])

    @property
    def is_polymer(self) -> np.ndarray:
        """Protein / nucleic acid atoms (ATOM records + modified residues like MSE)."""
        return (self.record == "ATOM") | np.isin(self.resname, ["MSE", "SEC", "PYL"])

    @property
    def is_metal(self) -> np.ndarray:
        return (self.record == "HETATM") & np.isin(self.element, list(METALS))

    def residue_keys(self) -> np.ndarray:
        """Per-atom residue identifier such as ``A:HIS57`` (insertion code appended)."""
        return np.array([
            f"{c or '_'}:{r}{s}{i}" for c, r, s, i in
            zip(self.chain, self.resname, self.resseq, self.icode)
        ])

    def subset(self, mask: np.ndarray) -> "Structure":
        return Structure(
            name=self.name, record=self.record[mask], serial=self.serial[mask],
            atom_name=self.atom_name[mask], resname=self.resname[mask],
            chain=self.chain[mask], resseq=self.resseq[mask], icode=self.icode[mask],
            coords=self.coords[mask], element=self.element[mask], title=self.title,
        )

    def ligands(self, include_additives: bool = False) -> list[dict]:
        """Distinct non-water, non-polymer HETATM groups (one entry per residue instance)."""
        mask = (self.record == "HETATM") & ~self.is_water & ~self.is_polymer
        seen: dict[tuple, dict] = {}
        for i in np.flatnonzero(mask):
            key = (str(self.chain[i]), str(self.resname[i]), int(self.resseq[i]), str(self.icode[i]))
            if key not in seen:
                seen[key] = {"chain": key[0], "resname": key[1], "resseq": key[2],
                             "icode": key[3], "n_atoms": 0}
            seen[key]["n_atoms"] += 1
        out = list(seen.values())
        if not include_additives:
            out = [l for l in out if l["resname"] not in COMMON_ADDITIVES
                   and not (l["n_atoms"] == 1 and l["resname"] in METALS)]
        return out


def _guess_element(atom_name: str, resname: str, record: str) -> str:
    name = atom_name.strip()
    if record == "HETATM" and name.upper() in METALS and resname.strip().upper() == name.upper():
        return name.upper()
    letters = re.sub(r"[^A-Za-z]", "", name)
    if not letters:
        return "X"
    # In PDB convention a 2-letter element starts in column 13 (atom name not space-padded).
    if atom_name[:1] != " " and len(letters) >= 2 and letters[:2].upper() in VDW_RADII \
            and record == "HETATM":
        return letters[:2].upper()
    return letters[0].upper()


def parse_pdb(text: str, name: str = "structure") -> Structure:
    """Parse PDB-format text. Keeps the first MODEL and the first alternate location."""
    rows = []
    title_parts = []
    kept_lines = []
    for line in text.splitlines():
        rec = line[:6].strip()
        if rec == "TITLE":
            title_parts.append(line[10:80].strip())
            continue
        if rec == "ENDMDL":
            break
        if rec not in ("ATOM", "HETATM"):
            continue
        line = line.ljust(80)
        altloc = line[16]
        if altloc not in (" ", "A", "1"):
            continue
        try:
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            resseq = int(line[22:26])
        except ValueError:
            continue
        atom_name = line[12:16]
        resname = line[17:20].strip()
        element = line[76:78].strip().upper() or _guess_element(atom_name, resname, rec)
        # Atoms are renumbered 1..N so serials are unique and match the viewer.
        serial = len(rows) + 1
        rows.append((rec, serial, atom_name.strip(), resname, line[21].strip(), resseq,
                     line[26].strip(), x, y, z, element))
        kept_lines.append(f"{rec:<6}{serial % 100000:5d}" + line[11:16] + " " + line[17:].rstrip())
    if not rows:
        raise ValueError(f"No ATOM/HETATM records found in {name!r}")
    cols = list(zip(*rows))
    return Structure(
        name=name,
        record=np.array(cols[0]), serial=np.array(cols[1], dtype=int),
        atom_name=np.array(cols[2]), resname=np.array(cols[3]),
        chain=np.array(cols[4]), resseq=np.array(cols[5], dtype=int),
        icode=np.array(cols[6]),
        coords=np.column_stack([cols[7], cols[8], cols[9]]).astype(float),
        element=np.array(cols[10]),
        pdb_text="\n".join(kept_lines) + "\nEND\n",
        title=" ".join(title_parts),
    )


def fetch_pdb(pdb_id: str, timeout: float = 30.0) -> str:
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 (fixed https host)
        return resp.read().decode("utf-8", errors="replace")


def load_structure(source: str) -> Structure:
    """Load from a local .pdb/.ent file, or fetch a 4-character PDB ID from RCSB."""
    if os.path.exists(source):
        with open(source, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        return parse_pdb(text, name=os.path.splitext(os.path.basename(source))[0])
    if _PDB_ID.match(source):
        return parse_pdb(fetch_pdb(source), name=source.upper())
    raise FileNotFoundError(f"{source!r} is neither an existing file nor a 4-character PDB ID")
