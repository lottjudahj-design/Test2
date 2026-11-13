import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr

import numpy as np

from pocketviz import cli
from pocketviz.geometry import connected_components, fibonacci_sphere
from pocketviz.pockets import detect_pockets
from pocketviz.sites import (SelectionError, find_interactions, ligand_sites, parse_ligand_spec,
                             parse_residue_spec, pocket_sites, residue_site, sphere_site)
from pocketviz.structure import parse_pdb


def read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def pdb_line(rec, serial, name, resn, chain, resi, xyz, elem, altloc=" "):
    return (f"{rec:<6}{serial:5d} {name:<4}{altloc}{resn:>3} {chain}{resi:4d}    "
            f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00          {elem:>2}")


def cavity_pdb(radius=6.5, n=500, opening=5.5, ligand=True):
    """A spherical shell of carbons (a 'protein' with an internal cavity) open at +z."""
    pts = fibonacci_sphere(n) * radius
    pts = pts[pts[:, 2] < opening]
    lines = []
    for i, p in enumerate(pts):
        lines.append(pdb_line("ATOM", i + 1, "CB", "LEU", "A", i // 4 + 1, p, "C"))
    if ligand:
        for j, (name, el, p) in enumerate([("C1", "C", (0, 0, 0)), ("C2", "C", (1.4, 0, 0)),
                                           ("N1", "N", (-1.3, 0, 0))]):
            lines.append(pdb_line("HETATM", len(pts) + j + 1, name, "LIG", "A", 900, p, el))
        lines.append(pdb_line("HETATM", len(pts) + 10, "O", "HOH", "A", 950, (0, 0, 20), "O"))
    return "\n".join(lines) + "\nEND\n"


class ParseTests(unittest.TestCase):
    def test_altloc_models_and_elements(self):
        text = "\n".join([
            "TITLE     TEST STRUCTURE",
            "MODEL        1",
            pdb_line("ATOM", 1, "CA", "GLY", "A", 1, (0, 0, 0), "C"),
            pdb_line("ATOM", 2, "CA", "SER", "A", 2, (1, 0, 0), "C", altloc="A"),
            pdb_line("ATOM", 3, "CA", "SER", "A", 2, (1.1, 0, 0), "C", altloc="B"),
            pdb_line("HETATM", 4, "ZN", "ZN", "A", 3, (5, 0, 0), "").replace("        ZN", ""),
            "ENDMDL",
            "MODEL        2",
            pdb_line("ATOM", 1, "CA", "GLY", "A", 1, (9, 9, 9), "C"),
            "ENDMDL",
        ])
        s = parse_pdb(text)
        self.assertEqual(len(s), 3)
        self.assertEqual(s.title, "TEST STRUCTURE")
        self.assertEqual(list(s.serial), [1, 2, 3])
        self.assertEqual(s.element[2], "ZN")
        self.assertTrue(s.is_metal[2])
        self.assertEqual(s.ligands(), [])  # lone metal ions are not auto-selected ligands
        self.assertEqual(len(s.ligands(include_additives=True)), 1)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            parse_pdb("HEADER nothing here\n")


class SpecTests(unittest.TestCase):
    def test_ligand_spec(self):
        self.assertEqual(parse_ligand_spec("ben"), {"resname": "BEN"})
        self.assertEqual(parse_ligand_spec("B:MK1:902"), {"chain": "B", "resname": "MK1", "resseq": 902})
        with self.assertRaises(SelectionError):
            parse_ligand_spec("a:b:1:2")

    def test_residue_spec(self):
        self.assertEqual(parse_residue_spec("A:57, 102,B:189-195"),
                         [("A", 57, 57), (None, 102, 102), ("B", 189, 195)])
        with self.assertRaises(SelectionError):
            parse_residue_spec("A57")


class GeometryTests(unittest.TestCase):
    def test_connected_components(self):
        pts = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [10, 10, 10], [11, 10, 10]])
        comps = connected_components(pts)
        self.assertEqual([len(c) for c in comps], [3, 2])


class DetectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s = parse_pdb(cavity_pdb(), name="cavity")
        cls.res = detect_pockets(cls.s)

    def test_cavity_found_at_center(self):
        self.assertGreaterEqual(len(self.res.pockets), 1)
        top = self.res.pockets[0]
        self.assertLess(np.linalg.norm(top.center), 1.5)
        self.assertGreater(top.mean_buriedness, 0.8)
        self.assertGreater(len(top.residues), 5)

    def test_no_pocket_on_flat_sheet(self):
        g = np.arange(-8, 8.1, 1.5)
        xy = np.array([(x, y, 0.0) for x in g for y in g])
        text = "\n".join(pdb_line("ATOM", i + 1, "CB", "ALA", "A", i + 1, p, "C") for i, p in enumerate(xy))
        self.assertEqual(detect_pockets(parse_pdb(text)).pockets, [])

    def test_ligand_site(self):
        (site,) = ligand_sites(self.s, "LIG", self.res, cutoff=5.5)
        self.assertEqual(site.kind, "ligand")
        self.assertEqual(site.overlapping_pockets, [1])
        self.assertEqual(len(site.ligand_atoms), 3)
        self.assertGreater(len(site.residues), 0)
        with self.assertRaises(SelectionError):
            ligand_sites(self.s, "NOPE", self.res)

    def test_residue_pocket_sphere_sites(self):
        self.assertEqual(len(residue_site(self.s, "A:1-5", self.res).residues), 5)
        with self.assertRaises(SelectionError):
            residue_site(self.s, "Z:1", self.res)
        (p,) = pocket_sites(self.s, [1], self.res)
        self.assertEqual(p.volume, self.res.pockets[0].volume)
        with self.assertRaises(SelectionError):
            pocket_sites(self.s, [99], self.res)
        sph = sphere_site(self.s, np.zeros(3), 7.0, self.res)
        self.assertGreater(len(sph.points), 0)


class InteractionTests(unittest.TestCase):
    def test_salt_bridge_hbond_hydrophobic(self):
        text = "\n".join([
            pdb_line("ATOM", 1, "OD1", "ASP", "A", 1, (3.0, 0, 0), "O"),
            pdb_line("ATOM", 2, "OG", "SER", "A", 2, (0, -2.9, 0), "O"),
            pdb_line("ATOM", 3, "CD1", "LEU", "A", 3, (5.0, 3.8, 0), "C"),
            pdb_line("ATOM", 4, "CA", "LEU", "A", 3, (5.0, 4.5, 0), "C"),
            pdb_line("HETATM", 5, "N1", "LIG", "A", 9, (0, 0, 0), "N"),
            pdb_line("HETATM", 6, "C1", "LIG", "A", 9, (5.0, 0, 0), "C"),
        ])
        s = parse_pdb(text)
        its = find_interactions(s, np.array([4, 5]), np.array([0, 1, 2, 3]))
        kinds = [(i["type"], i["residue"]) for i in its]
        self.assertIn(("salt bridge", "A:ASP1"), kinds)
        self.assertIn(("hydrogen bond", "A:SER2"), kinds)
        self.assertIn(("hydrophobic", "A:LEU3"), kinds)
        self.assertEqual(sum(k == "hydrophobic" for k, _ in kinds), 1)  # one per residue, no backbone


class CliTests(unittest.TestCase):
    def test_end_to_end(self):
        with tempfile.TemporaryDirectory() as d:
            pdb = os.path.join(d, "cav.pdb")
            with open(pdb, "w") as fh:
                fh.write(cavity_pdb())
            out, js, cs = (os.path.join(d, f) for f in ("o.html", "o.json", "o.csv"))
            with redirect_stderr(io.StringIO()):
                rc = cli.main([pdb, "--residues", "A:1-3", "--name", "Mine", "-o", out,
                               "--json", js, "--csv", cs])
            self.assertEqual(rc, 0)
            html = read(out)
            self.assertIn("3Dmol", html)
            self.assertEqual(html.count("</script>"), 2)  # data cannot break out of its tag
            data = json.loads(read(js))
            self.assertEqual([s["name"] for s in data["sites"]], ["Mine"])
            self.assertGreater(len(read(cs).splitlines()), 1)

            with redirect_stderr(io.StringIO()):  # default: ligands + top pockets not already covered
                cli.main([pdb, "-o", out, "--json", js])
            kinds = [s["kind"] for s in json.loads(read(js))["sites"]]
            self.assertEqual(kinds[0], "ligand")

            err = io.StringIO()
            with redirect_stderr(err):
                self.assertEqual(cli.main([pdb, "--ligand", "XXX", "-o", out]), 2)
            self.assertIn("Available: A:LIG:900", err.getvalue())


if __name__ == "__main__":
    unittest.main()
