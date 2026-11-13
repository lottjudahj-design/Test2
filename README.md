# pocketviz

Visualise the ligand-binding sites you care about in a protein structure.

`pocketviz` reads a PDB file (or downloads one by PDB ID), finds buried pockets
on the protein surface, builds the binding sites you ask for, and writes a
single interactive HTML page. The page uses [3Dmol.js](https://3dmol.csb.pitt.edu/)
and opens in any browser.

The only dependency is `numpy`.

## Install

```bash
pip install -e .          # installs the `pocketviz` command
# or run without installing:
python -m pocketviz --help
```

## Choosing sites

Every selection flag can be repeated and combined. Each site gets its own color.

| You want | Flag | Example |
|---|---|---|
| The site around a bound ligand | `--ligand RES`, `CHAIN:RES` or `CHAIN:RES:NUM` | `--ligand BEN`, `--ligand B:MK1:902` |
| A site defined by residues you know | `--residues LIST` (+ optional `--name`) | `--residues A:57,A:102,A:189-195 --name "S1 pocket"` |
| A pocket from the detected ranking | `--pocket IDS` | `--pocket 1,3` |
| Everything around a coordinate | `--center X,Y,Z --radius R` | `--center 12.1,4.5,-3 --radius 8` |
| No flag | bound ligands (no water, buffer molecules or lone ions), plus the top `--top N` (default 3) pockets they don't already cover | |

```bash
pocketviz 3PTB                                   # trypsin + benzamidine, auto mode
pocketviz 3PTB --ligand BEN --open               # open the page when done
pocketviz 1HSG --ligand MK1 --residues A:25,B:25 --name "Catalytic Asp dyad"
pocketviz my_model.pdb --pocket 1,2 --json report.json --csv residues.csv
```

The command prints a summary to stderr:

```
Loaded 3ptb: 1701 atoms
Detected 5 pockets
  [ligand] BEN A:1: 13 residues, 44 Å³, pocket 1, 9 contacts
  [pocket] Pocket 2: 19 residues, 59 Å³, pocket 2
  ...
Wrote 3ptb_sites.html
```

## The viewer

- **Protein:** cartoon, surface, cartoon plus a translucent surface, or lines.
- **Each site:** lining residues as sticks in the site color, the ligand in green
  ball-and-stick, and the pocket volume as probe points or as a surface blob.
- **Interaction lines:** dashed lines for H-bonds (blue), salt bridges (red),
  hydrophobic contacts (grey) and metal coordination (purple).
- **Site surface:** a surface over the site, colored by the site color or by
  residue property (hydrophobic, aromatic, polar, positive, negative).
- **Sidebar:** a card per site with show/hide and focus buttons, a residue
  composition bar, a residue table (click a row to zoom to it) and an
  interaction table. Below that is the full ranked pocket list; tick any pocket
  to overlay it.
- **Also:** atom info on hover, labels for the focused site, a dark background
  toggle and PNG export.

The page loads 3Dmol.js from a CDN, so it needs internet access when opened.
The structure and site data are embedded in the page.

## How pockets are found

This is a LIGSITE-style grid method:

1. Put a grid (default 1 Å) around the receptor, meaning protein and metal ions.
   Water and ligands are excluded, so occupied sites still show up as pockets.
2. Keep grid points where a probe sphere (default 1.4 Å) fits without touching
   any van der Waals sphere, and that lie within 6 Å of the surface.
3. Cast 30 rays from each point. A point is a pocket point if at least 60% of
   the rays hit protein within 12 Å.
4. Group neighboring pocket points into clusters (26-connectivity). Drop
   clusters smaller than 30 Å³. Rank the rest by `volume × mean_buriedness²`.

All thresholds are flags: `--spacing`, `--probe`, `--min-buriedness`,
`--min-volume`, `--max-pockets`.

On a quick check, the crystallographic ligand sits in pocket #1 for 3PTB
(trypsin), 1HSG (HIV-1 protease) and 1STP (streptavidin), and in pocket #2 for
3EML (A2A receptor).

## Limitations

- Only PDB format is read, not mmCIF. Only the first model and the first
  alternate location are kept.
- Interactions are distance-only candidates. Protonation states and angles are
  not checked, so treat them as a guide.
- The pocket score is a geometric heuristic. It is not a druggability prediction.

## Python API

```python
from pocketviz import load_structure, detect_pockets, ligand_sites, residue_site, write_html
from pocketviz.sites import assign_colors

s = load_structure("3PTB")
pockets = detect_pockets(s)
sites = ligand_sites(s, "BEN", pockets) + [residue_site(s, "A:57,A:102,A:195", pockets, name="Catalytic triad")]
assign_colors(sites)
write_html("trypsin.html", s, sites, pockets)
```

## Tests

```bash
python -m unittest discover -s tests
```
