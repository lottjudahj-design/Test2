"""Geometric pocket detection (LIGSITE-style buriedness on a grid)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import Grid, connected_components, min_distances, ray_buriedness, surface_distance_map
from .structure import Structure


@dataclass
class Pocket:
    id: int
    points: np.ndarray          # (n, 3) grid point coordinates, Å
    buriedness: np.ndarray      # (n,) fraction of enclosed directions
    volume: float               # Å^3
    score: float
    lining_atoms: np.ndarray    # indices into the full Structure
    residues: list[str] = field(default_factory=list)

    @property
    def center(self) -> np.ndarray:
        return self.points.mean(axis=0)

    @property
    def mean_buriedness(self) -> float:
        return float(self.buriedness.mean())


@dataclass
class PocketResult:
    pockets: list[Pocket]
    candidate_points: np.ndarray      # all buried probe points (any cluster size)
    candidate_buriedness: np.ndarray
    params: dict


def receptor_mask(struct: Structure) -> np.ndarray:
    """Atoms that form the pocket walls: polymer + metals, no hydrogens, no water/ligands."""
    return (struct.is_polymer | struct.is_metal) & ~struct.is_hydrogen


def detect_pockets(struct: Structure, spacing: float = 1.0, probe: float = 1.4,
                   min_buriedness: float = 0.6, ray_length: float = 12.0, n_dirs: int = 30,
                   min_volume: float = 30.0, max_pockets: int = 20,
                   max_surface_distance: float = 6.0) -> PocketResult:
    """Find buried empty regions of the receptor.

    A grid point is a pocket candidate when a probe sphere of radius ``probe``
    fits there without touching any receptor van der Waals sphere, it lies
    within ``max_surface_distance`` of the receptor surface, and at least
    ``min_buriedness`` of ``n_dirs`` rays cast from it hit the receptor within
    ``ray_length``. Candidates are clustered by 26-connectivity and clusters are
    ranked by ``volume * mean_buriedness**2``.
    """
    rec = receptor_mask(struct)
    coords, radii = struct.coords[rec], struct.radii[rec]
    rec_idx = np.flatnonzero(rec)
    grid = Grid.around(coords, margin=max_surface_distance + 2.0, spacing=spacing)
    sdist = surface_distance_map(grid, coords, radii, cutoff=max_surface_distance + 1.0)
    occupied = sdist < 0
    free = (sdist >= probe) & (sdist <= max_surface_distance)
    ijk = np.argwhere(free)
    bur = ray_buriedness(occupied, ijk, n_dirs=n_dirs,
                         max_steps=int(round(ray_length / spacing)))
    keep = bur >= min_buriedness
    ijk, bur = ijk[keep], bur[keep]
    xyz = grid.to_xyz(ijk)

    voxel = spacing ** 3
    pockets: list[Pocket] = []
    for members in connected_components(ijk):
        vol = len(members) * voxel
        if vol < min_volume:
            break  # clusters are sorted largest first
        pts, b = xyz[members], bur[members]
        pockets.append(Pocket(id=0, points=pts, buriedness=b, volume=vol,
                              score=vol * float(b.mean()) ** 2, lining_atoms=np.array([], int)))
    pockets.sort(key=lambda p: p.score, reverse=True)
    pockets = pockets[:max_pockets]

    keys = struct.residue_keys()
    lining_cut = probe + spacing + 0.5
    for n, p in enumerate(pockets, start=1):
        p.id = n
        d, _ = min_distances(coords, p.points)
        near = (d - radii) <= lining_cut
        p.lining_atoms = rec_idx[near]
        p.residues = list(dict.fromkeys(keys[p.lining_atoms]))
    return PocketResult(pockets=pockets, candidate_points=xyz, candidate_buriedness=bur,
                        params=dict(spacing=spacing, probe=probe, min_buriedness=min_buriedness,
                                    ray_length=ray_length, n_dirs=n_dirs, min_volume=min_volume,
                                    max_surface_distance=max_surface_distance))
