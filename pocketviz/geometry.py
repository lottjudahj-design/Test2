"""Grid utilities: distance-to-surface maps, ray-cast buriedness, clustering."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


def fibonacci_sphere(n: int) -> np.ndarray:
    """Return ``n`` approximately uniformly distributed unit vectors."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    return np.column_stack([np.cos(theta) * np.sin(phi),
                            np.sin(theta) * np.sin(phi),
                            np.cos(phi)])


@dataclass
class Grid:
    origin: np.ndarray
    spacing: float
    shape: tuple[int, int, int]

    @classmethod
    def around(cls, coords: np.ndarray, margin: float, spacing: float) -> "Grid":
        lo = coords.min(axis=0) - margin
        hi = coords.max(axis=0) + margin
        shape = tuple(int(n) for n in np.ceil((hi - lo) / spacing).astype(int) + 1)
        return cls(origin=lo, spacing=spacing, shape=shape)

    @property
    def size(self) -> int:
        return int(np.prod(self.shape))

    def to_xyz(self, ijk: np.ndarray) -> np.ndarray:
        return self.origin + np.asarray(ijk) * self.spacing

    def to_ijk(self, xyz: np.ndarray) -> np.ndarray:
        return np.rint((np.asarray(xyz) - self.origin) / self.spacing).astype(int)

    def in_bounds(self, ijk: np.ndarray) -> np.ndarray:
        return np.all((ijk >= 0) & (ijk < np.array(self.shape)), axis=-1)


def surface_distance_map(grid: Grid, coords: np.ndarray, radii: np.ndarray,
                         cutoff: float) -> np.ndarray:
    """For every grid point: min over atoms of (|p - x_a| - r_a), capped at ``cutoff``.

    Negative values are inside an atom's van der Waals sphere. Computed by
    stamping a cube of offsets around every atom, which is O(atoms * reach^3)
    and needs nothing beyond numpy.
    """
    dist = np.full(grid.size, cutoff, dtype=np.float32)
    reach = int(np.ceil((cutoff + radii.max()) / grid.spacing))
    base = grid.to_ijk(coords)
    rng = np.arange(-reach, reach + 1)
    offsets = np.stack(np.meshgrid(rng, rng, rng, indexing="ij"), -1).reshape(-1, 3)
    offsets = offsets[np.linalg.norm(offsets, axis=1) <= reach + 1]
    strides = np.array([grid.shape[1] * grid.shape[2], grid.shape[2], 1])
    for off in offsets:
        ijk = base + off
        ok = grid.in_bounds(ijk)
        if not ok.any():
            continue
        ijk = ijk[ok]
        d = np.linalg.norm(grid.to_xyz(ijk) - coords[ok], axis=1) - radii[ok]
        np.minimum.at(dist, ijk @ strides, d.astype(np.float32))
    return dist.reshape(grid.shape)


def ray_buriedness(occupied: np.ndarray, points_ijk: np.ndarray, n_dirs: int,
                   max_steps: int) -> np.ndarray:
    """Fraction of ``n_dirs`` rays from each point that hit an occupied voxel
    within ``max_steps`` grid steps. 1.0 = fully enclosed, ~0 = solvent exposed."""
    dirs = fibonacci_sphere(n_dirs)
    shape = np.array(occupied.shape)
    hits = np.zeros(len(points_ijk), dtype=np.int32)
    pts = points_ijk.astype(float)
    for d in dirs:
        hit = np.zeros(len(points_ijk), dtype=bool)
        for s in range(1, max_steps + 1):
            q = np.rint(pts + s * d).astype(int)
            ok = np.all((q >= 0) & (q < shape), axis=1) & ~hit
            if not ok.any():
                break
            idx = np.flatnonzero(ok)
            hit[idx] = occupied[q[idx, 0], q[idx, 1], q[idx, 2]]
        hits += hit
    return hits / n_dirs


def connected_components(points_ijk: np.ndarray) -> list[np.ndarray]:
    """Group grid points into 26-connected clusters. Returns index arrays, largest first."""
    lookup = {tuple(p): i for i, p in enumerate(points_ijk.tolist())}
    rng = (-1, 0, 1)
    neigh = [(a, b, c) for a in rng for b in rng for c in rng if (a, b, c) != (0, 0, 0)]
    label = np.full(len(points_ijk), -1)
    clusters = []
    for start in range(len(points_ijk)):
        if label[start] >= 0:
            continue
        label[start] = len(clusters)
        members = [start]
        queue = deque([start])
        while queue:
            x, y, z = points_ijk[queue.popleft()]
            for a, b, c in neigh:
                j = lookup.get((x + a, y + b, z + c))
                if j is not None and label[j] < 0:
                    label[j] = len(clusters)
                    members.append(j)
                    queue.append(j)
        clusters.append(np.array(members))
    clusters.sort(key=len, reverse=True)
    return clusters


def min_distances(a: np.ndarray, b: np.ndarray, chunk: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """For each point in ``a``: distance to and index of the nearest point in ``b``."""
    if len(b) == 0:
        return np.full(len(a), np.inf), np.full(len(a), -1)
    best = np.empty(len(a))
    arg = np.empty(len(a), dtype=int)
    for s in range(0, len(a), chunk):
        d = np.linalg.norm(a[s:s + chunk, None, :] - b[None, :, :], axis=2)
        arg[s:s + chunk] = d.argmin(axis=1)
        best[s:s + chunk] = d[np.arange(len(d)), arg[s:s + chunk]]
    return best, arg
