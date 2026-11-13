"""Select the binding sites a user cares about and characterise them."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from .geometry import min_distances
from .pockets import PocketResult, receptor_mask
from .structure import Structure

PALETTE = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4",
           "#f032e6", "#9a6324", "#469990", "#808000", "#000075", "#bfef45"]

RESIDUE_CLASS = {
    **dict.fromkeys(["ALA", "VAL", "LEU", "ILE", "MET", "PRO", "GLY", "MSE"], "hydrophobic"),
    **dict.fromkeys(["PHE", "TRP", "TYR"], "aromatic"),
    **dict.fromkeys(["SER", "THR", "ASN", "GLN", "CYS"], "polar"),
    **dict.fromkeys(["LYS", "ARG", "HIS"], "positive"),
    **dict.fromkeys(["ASP", "GLU"], "negative"),
}
HYDROPHOBIC_RES = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO", "TYR", "CYS", "MSE"}
ANIONIC_ATOMS = {("ASP", "OD1"), ("ASP", "OD2"), ("GLU", "OE1"), ("GLU", "OE2")}
CATIONIC_ATOMS = {("LYS", "NZ"), ("ARG", "NE"), ("ARG", "NH1"), ("ARG", "NH2"),
                  ("HIS", "ND1"), ("HIS", "NE2")}
BACKBONE = {"N", "CA", "C", "O"}


@dataclass
class Site:
    name: str
    kind: str                       # ligand | pocket | residues | sphere
    color: str
    description: str
    site_atoms: np.ndarray          # receptor atom indices lining the site
    ligand_atoms: np.ndarray = field(default_factory=lambda: np.array([], int))
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    residues: list[dict] = field(default_factory=list)
    interactions: list[dict] = field(default_factory=list)
    overlapping_pockets: list[int] = field(default_factory=list)
    volume: float | None = None
    buriedness: float | None = None

    @property
    def center(self) -> np.ndarray:
        if len(self.points):
            return self.points.mean(axis=0)
        return np.zeros(3)


class SelectionError(ValueError):
    pass


# --------------------------------------------------------------------------- parsing
def parse_ligand_spec(spec: str) -> dict:
    """``RES``, ``CHAIN:RES`` or ``CHAIN:RES:NUM`` (e.g. ``BEN``, ``A:MK1:902``)."""
    parts = [p.strip() for p in spec.split(":")]
    if len(parts) == 1:
        return {"resname": parts[0].upper()}
    if len(parts) == 2:
        return {"chain": parts[0], "resname": parts[1].upper()}
    if len(parts) == 3:
        return {"chain": parts[0], "resname": parts[1].upper(), "resseq": int(parts[2])}
    raise SelectionError(f"Bad ligand spec {spec!r}; use RES, CHAIN:RES or CHAIN:RES:NUM")


_RES_TOKEN = re.compile(r"^(?:(?P<chain>[A-Za-z0-9]):)?(?P<start>-?\d+)(?:-(?P<end>-?\d+))?$")


def parse_residue_spec(spec: str) -> list[tuple[str | None, int, int]]:
    """``A:57,A:102,A:189-195`` or ``57,102`` -> [(chain|None, start, end), ...]."""
    out = []
    for tok in filter(None, (t.strip() for t in spec.split(","))):
        m = _RES_TOKEN.match(tok)
        if not m:
            raise SelectionError(f"Bad residue token {tok!r}; use CHAIN:NUM, NUM or CHAIN:START-END")
        start = int(m["start"])
        out.append((m["chain"], start, int(m["end"]) if m["end"] else start))
    return out


def parse_xyz(spec: str) -> np.ndarray:
    vals = [float(v) for v in spec.replace(" ", "").split(",")]
    if len(vals) != 3:
        raise SelectionError(f"Center must be x,y,z; got {spec!r}")
    return np.array(vals)


# --------------------------------------------------------------------------- helpers
def _points_near(points: np.ndarray, atoms_xyz: np.ndarray, cutoff: float) -> np.ndarray:
    if len(points) == 0 or len(atoms_xyz) == 0:
        return np.zeros((0, 3))
    d, _ = min_distances(points, atoms_xyz)
    return points[d <= cutoff]


def _residue_table(struct: Structure, atom_idx: np.ndarray, ref_xyz: np.ndarray) -> list[dict]:
    keys = struct.residue_keys()
    rows: dict[str, dict] = {}
    d = min_distances(struct.coords[atom_idx], ref_xyz)[0] if len(ref_xyz) else np.zeros(len(atom_idx))
    for i, dist in zip(atom_idx, d):
        k = keys[i]
        if k not in rows:
            rn = str(struct.resname[i])
            rows[k] = {"key": k, "chain": str(struct.chain[i]), "resname": rn,
                       "resseq": int(struct.resseq[i]), "icode": str(struct.icode[i]),
                       "class": RESIDUE_CLASS.get(rn, "other"), "min_dist": float(dist),
                       "atoms": []}
        rows[k]["min_dist"] = min(rows[k]["min_dist"], float(dist))
        rows[k]["atoms"].append(int(struct.serial[i]))
    return sorted(rows.values(), key=lambda r: (r["chain"], r["resseq"], r["icode"]))


def _overlaps(points: np.ndarray, pockets: PocketResult, cutoff: float = 2.0) -> list[int]:
    out = []
    for p in pockets.pockets:
        if len(points) and min_distances(points, p.points)[0].min() <= cutoff:
            out.append(p.id)
    return out


def find_interactions(struct: Structure, lig_idx: np.ndarray, rec_idx: np.ndarray) -> list[dict]:
    """Distance-based interaction candidates between a ligand and the receptor.

    These are geometric heuristics (no protonation states or angles), meant to
    guide the eye: H-bond (N/O–N/O ≤ 3.5 Å), salt bridge (charged side-chain
    atom to ligand N/O ≤ 4.0 Å), hydrophobic (C–apolar side-chain C ≤ 4.0 Å,
    closest pair per residue) and metal coordination (≤ 2.8 Å).
    """
    out = []
    keys = struct.residue_keys()
    el, names, resn = struct.element, struct.atom_name, struct.resname
    lx = struct.coords[lig_idx]
    best_hphob: dict[str, tuple[float, int, int]] = {}
    for r in rec_idx:
        d = np.linalg.norm(lx - struct.coords[r], axis=1)
        for li, dist in zip(lig_idx, d):
            if dist > 4.0:
                continue
            le, re_, pair = el[li], el[r], (str(resn[r]), str(names[r]))
            kind = None
            if struct.is_metal[r] and le in ("N", "O", "S") and dist <= 2.8:
                kind = "metal"
            elif le in ("N", "O") and (pair in ANIONIC_ATOMS or pair in CATIONIC_ATOMS) \
                    and (pair in CATIONIC_ATOMS) == (le == "O"):
                kind = "salt bridge"
            elif le in ("N", "O") and re_ in ("N", "O") and dist <= 3.5:
                kind = "hydrogen bond"
            elif le in ("C", "CL", "BR", "I", "F", "S") and re_ == "C" \
                    and resn[r] in HYDROPHOBIC_RES and names[r] not in BACKBONE:
                k = keys[r]
                if k not in best_hphob or dist < best_hphob[k][0]:
                    best_hphob[k] = (float(dist), int(li), int(r))
                continue
            if kind:
                out.append(_interaction(struct, kind, li, r, dist))
    for dist, li, r in best_hphob.values():
        out.append(_interaction(struct, "hydrophobic", li, r, dist))
    order = {"metal": 0, "salt bridge": 1, "hydrogen bond": 2, "hydrophobic": 3}
    out.sort(key=lambda x: (order[x["type"]], x["distance"]))
    return out


def _interaction(struct: Structure, kind: str, li: int, ri: int, dist: float) -> dict:
    return {"type": kind, "distance": round(float(dist), 2),
            "ligand_atom": f"{struct.resname[li]} {struct.atom_name[li]}",
            "residue": str(struct.residue_keys()[ri]), "residue_atom": str(struct.atom_name[ri]),
            "ligand_serial": int(struct.serial[li]), "residue_serial": int(struct.serial[ri]),
            "a": struct.coords[li].round(3).tolist(), "b": struct.coords[ri].round(3).tolist()}


# --------------------------------------------------------------------------- builders
def _finish(site: Site, struct: Structure, ref_xyz: np.ndarray, pockets: PocketResult) -> Site:
    site.residues = _residue_table(struct, site.site_atoms, ref_xyz)
    if site.volume is None and len(site.points):
        site.volume = len(site.points) * pockets.params["spacing"] ** 3
    if not site.overlapping_pockets:
        site.overlapping_pockets = _overlaps(site.points if len(site.points) else ref_xyz, pockets)
    return site


def ligand_sites(struct: Structure, spec: str, pockets: PocketResult, cutoff: float = 4.5) -> list[Site]:
    q = parse_ligand_spec(spec)
    matches = [l for l in struct.ligands(include_additives=True)
               if l["resname"] == q["resname"]
               and ("chain" not in q or l["chain"] == q["chain"])
               and ("resseq" not in q or l["resseq"] == q["resseq"])]
    if not matches:
        avail = ", ".join(sorted({f"{l['chain']}:{l['resname']}:{l['resseq']}"
                                  for l in struct.ligands(include_additives=True)})) or "none"
        raise SelectionError(f"No ligand matches {spec!r}. Available: {avail}")
    rec = np.flatnonzero(receptor_mask(struct))
    sites = []
    for l in matches:
        lig = np.flatnonzero((struct.chain == l["chain"]) & (struct.resname == l["resname"])
                             & (struct.resseq == l["resseq"]) & (struct.icode == l["icode"])
                             & ~struct.is_hydrogen)
        lx = struct.coords[lig]
        d, _ = min_distances(struct.coords[rec], lx)
        site_atoms = rec[d <= cutoff]
        pts = _points_near(pockets.candidate_points, lx, 2.5)
        site = Site(name=f"{l['resname']} {l['chain']}:{l['resseq']}{l['icode']}", kind="ligand",
                    color="", description=f"Residues within {cutoff} Å of ligand {l['resname']}",
                    site_atoms=site_atoms, ligand_atoms=lig, points=pts)
        site.interactions = find_interactions(struct, lig, site_atoms)
        site.overlapping_pockets = _overlaps(lx, pockets, cutoff=2.5)
        sites.append(_finish(site, struct, lx, pockets))
    return sites


def pocket_sites(struct: Structure, ids: list[int], pockets: PocketResult) -> list[Site]:
    by_id = {p.id: p for p in pockets.pockets}
    out = []
    for i in ids:
        if i not in by_id:
            raise SelectionError(f"Pocket {i} not found (detected: 1..{len(by_id)})")
        p = by_id[i]
        site = Site(name=f"Pocket {p.id}", kind="pocket", color="",
                    description=f"Detected pocket #{p.id}: {p.volume:.0f} Å³, "
                                f"buriedness {p.mean_buriedness:.2f}",
                    site_atoms=p.lining_atoms, points=p.points, overlapping_pockets=[p.id],
                    volume=p.volume, buriedness=p.mean_buriedness)
        out.append(_finish(site, struct, p.points, pockets))
    return out


def residue_site(struct: Structure, spec: str, pockets: PocketResult, name: str | None = None) -> Site:
    rec = receptor_mask(struct)
    mask = np.zeros(len(struct), bool)
    for chain, a, b in parse_residue_spec(spec):
        m = rec & (struct.resseq >= a) & (struct.resseq <= b)
        if chain is not None:
            m &= struct.chain == chain
        if not m.any():
            raise SelectionError(f"No receptor residues match {chain + ':' if chain else ''}{a}"
                                 f"{'-' + str(b) if b != a else ''}")
        mask |= m
    atoms = np.flatnonzero(mask)
    xyz = struct.coords[atoms]
    center = xyz.mean(axis=0)
    radius = float(np.linalg.norm(xyz - center, axis=1).max())
    pts = _points_near(pockets.candidate_points, xyz, pockets.params["probe"] + 3.0)
    pts = pts[np.linalg.norm(pts - center, axis=1) <= radius] if len(pts) else pts
    site = Site(name=name or f"Residues {spec}", kind="residues", color="",
                description=f"User-defined residues {spec}", site_atoms=atoms, points=pts)
    return _finish(site, struct, pts if len(pts) else xyz, pockets)


def sphere_site(struct: Structure, center: np.ndarray, radius: float, pockets: PocketResult) -> Site:
    rec = np.flatnonzero(receptor_mask(struct))
    d = np.linalg.norm(struct.coords[rec] - center, axis=1)
    pts = pockets.candidate_points
    pts = pts[np.linalg.norm(pts - center, axis=1) <= radius] if len(pts) else pts
    site = Site(name=f"Sphere ({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f}) r={radius:g}",
                kind="sphere", color="",
                description=f"Receptor atoms within {radius:g} Å of a point",
                site_atoms=rec[d <= radius], points=pts)
    if not len(site.site_atoms):
        raise SelectionError("No receptor atoms within the requested sphere")
    return _finish(site, struct, center[None, :], pockets)


def assign_colors(sites: list[Site]) -> None:
    for i, s in enumerate(sites):
        s.color = PALETTE[i % len(PALETTE)]
