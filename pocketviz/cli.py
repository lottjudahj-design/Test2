"""Command-line entry point: ``pocketviz STRUCTURE [selections] -o sites.html``."""
from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

from .pockets import detect_pockets
from .render import write_csv, write_html, write_json
from .sites import (SelectionError, assign_colors, ligand_sites, parse_xyz, pocket_sites,
                    residue_site, sphere_site)
from .structure import load_structure


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="pocketviz",
        description="Detect ligand-binding pockets and visualise the sites you choose in an "
                    "interactive 3D HTML viewer.",
        epilog="Examples:\n"
               "  pocketviz 3PTB                         # auto: bound ligands + top 3 pockets\n"
               "  pocketviz 3PTB --ligand BEN            # site around benzamidine\n"
               "  pocketviz 1HSG --residues A:25,B:25,A:48-50 --name 'Catalytic site'\n"
               "  pocketviz my.pdb --pocket 1,2 --center 10,4.5,-3 --radius 8\n",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("structure", help="PDB file path or 4-character PDB ID (downloaded from RCSB)")
    sel = ap.add_argument_group("site selection (repeatable; combine freely)")
    sel.add_argument("--ligand", action="append", default=[], metavar="SPEC",
                     help="ligand by RES, CHAIN:RES or CHAIN:RES:NUM, e.g. BEN or B:MK1:902")
    sel.add_argument("--residues", action="append", default=[], metavar="LIST",
                     help="residue list, e.g. A:57,A:102,A:189-195 (chain optional)")
    sel.add_argument("--name", action="append", default=[],
                     help="name for the matching --residues site (in order)")
    sel.add_argument("--pocket", action="append", default=[], metavar="IDS",
                     help="detected pocket rank(s), e.g. 1 or 1,3")
    sel.add_argument("--center", action="append", default=[], metavar="X,Y,Z",
                     help="site centred on a coordinate (use with --radius)")
    sel.add_argument("--radius", type=float, default=8.0, help="radius for --center sites (Å, default 8)")
    sel.add_argument("--top", type=int, default=3,
                     help="when nothing is selected: show bound ligands plus this many top pockets (default 3)")
    sel.add_argument("--cutoff", type=float, default=4.5,
                     help="ligand-to-residue distance defining a ligand site (Å, default 4.5)")
    det = ap.add_argument_group("pocket detection")
    det.add_argument("--spacing", type=float, default=1.0, help="grid spacing Å (default 1.0)")
    det.add_argument("--probe", type=float, default=1.4, help="probe radius Å (default 1.4)")
    det.add_argument("--min-buriedness", type=float, default=0.6,
                     help="fraction of rays that must hit protein (0-1, default 0.6)")
    det.add_argument("--min-volume", type=float, default=30.0, help="min pocket volume Å³ (default 30)")
    det.add_argument("--max-pockets", type=int, default=20, help="keep at most N pockets (default 20)")
    out = ap.add_argument_group("output")
    out.add_argument("-o", "--output", help="HTML file (default: <name>_sites.html)")
    out.add_argument("--json", help="also write a JSON report")
    out.add_argument("--csv", help="also write a per-residue CSV table")
    out.add_argument("--open", action="store_true", help="open the HTML in a web browser")
    out.add_argument("-q", "--quiet", action="store_true", help="no summary on stdout")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    log = (lambda *a: None) if args.quiet else (lambda *a: print(*a, file=sys.stderr))
    try:
        struct = load_structure(args.structure)
    except (OSError, ValueError) as e:
        print(f"error: could not load {args.structure!r}: {e}", file=sys.stderr)
        return 2
    log(f"Loaded {struct.name}: {len(struct)} atoms")
    pockets = detect_pockets(struct, spacing=args.spacing, probe=args.probe,
                             min_buriedness=args.min_buriedness, min_volume=args.min_volume,
                             max_pockets=args.max_pockets)
    log(f"Detected {len(pockets.pockets)} pockets")

    sites = []
    try:
        for spec in args.ligand:
            sites += ligand_sites(struct, spec, pockets, cutoff=args.cutoff)
        for i, spec in enumerate(args.residues):
            sites.append(residue_site(struct, spec, pockets,
                                      name=args.name[i] if i < len(args.name) else None))
        ids = [int(x) for spec in args.pocket for x in spec.split(",") if x.strip()]
        sites += pocket_sites(struct, ids, pockets)
        for spec in args.center:
            sites.append(sphere_site(struct, parse_xyz(spec), args.radius, pockets))
        if not sites:
            for lig in struct.ligands():
                sites += ligand_sites(struct, f"{lig['chain']}:{lig['resname']}:{lig['resseq']}",
                                      pockets, cutoff=args.cutoff)
            covered = {i for s in sites for i in s.overlapping_pockets}
            extra = [p.id for p in pockets.pockets if p.id not in covered][:args.top]
            sites += pocket_sites(struct, extra, pockets)
    except (SelectionError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    assign_colors(sites)

    for s in sites:
        extra = f", {s.volume:.0f} Å³" if s.volume else ""
        pk = f", pocket {','.join(map(str, s.overlapping_pockets))}" if s.overlapping_pockets else ""
        log(f"  [{s.kind}] {s.name}: {len(s.residues)} residues{extra}{pk}"
            + (f", {len(s.interactions)} contacts" if s.interactions else ""))

    html_path = Path(args.output or f"{struct.name}_sites.html")
    write_html(html_path, struct, sites, pockets)
    log(f"Wrote {html_path}")
    if args.json:
        log(f"Wrote {write_json(args.json, struct, sites, pockets)}")
    if args.csv:
        log(f"Wrote {write_csv(args.csv, sites)}")
    if args.open:
        webbrowser.open(html_path.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
