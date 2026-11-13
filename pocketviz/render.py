"""Write the interactive HTML viewer (3Dmol.js) and machine-readable reports."""
from __future__ import annotations

import csv
import html
import json
from pathlib import Path

import numpy as np

from .pockets import PocketResult
from .sites import Site
from .structure import Structure

THREEDMOL_URL = "https://cdn.jsdelivr.net/npm/3dmol@2.4.0/build/3Dmol-min.js"


def _site_payload(struct: Structure, s: Site) -> dict:
    return {
        "name": s.name, "kind": s.kind, "color": s.color, "description": s.description,
        "residues": [{k: v for k, v in r.items()} | {"min_dist": round(r["min_dist"], 2)}
                     for r in s.residues],
        "ligand_serials": [int(struct.serial[i]) for i in s.ligand_atoms],
        "points": np.round(s.points, 2).tolist(),
        "interactions": s.interactions,
        "overlapping_pockets": s.overlapping_pockets,
        "volume": None if s.volume is None else round(float(s.volume), 1),
        "buriedness": None if s.buriedness is None else round(float(s.buriedness), 3),
        "center": np.round(s.center, 2).tolist() if len(s.points) else None,
    }


def build_payload(struct: Structure, sites: list[Site], pockets: PocketResult) -> dict:
    return {
        "name": struct.name, "title": struct.title, "pdb": struct.pdb_text,
        "n_atoms": len(struct),
        "sites": [_site_payload(struct, s) for s in sites],
        "pockets": [{"id": p.id, "volume": round(p.volume, 1),
                     "buriedness": round(p.mean_buriedness, 3), "score": round(p.score, 1),
                     "n_residues": len(p.residues), "residues": p.residues,
                     "center": np.round(p.center, 2).tolist(),
                     "points": np.round(p.points, 2).tolist()} for p in pockets.pockets],
        "params": pockets.params,
    }


def write_html(path: str | Path, struct: Structure, sites: list[Site], pockets: PocketResult) -> Path:
    payload = json.dumps(build_payload(struct, sites, pockets), separators=(",", ":"))
    payload = payload.replace("</", "<\\/")  # never close the <script> early
    doc = (_TEMPLATE.replace("__TITLE__", html.escape(f"{struct.name} binding sites"))
           .replace("__3DMOL__", THREEDMOL_URL).replace("__DATA__", payload))
    path = Path(path)
    path.write_text(doc, encoding="utf-8")
    return path


def write_json(path: str | Path, struct: Structure, sites: list[Site], pockets: PocketResult) -> Path:
    data = build_payload(struct, sites, pockets)
    data.pop("pdb")
    path = Path(path)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def write_csv(path: str | Path, sites: list[Site]) -> Path:
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["site", "kind", "chain", "resname", "resseq", "icode", "class",
                    "min_dist_A", "interactions"])
        for s in sites:
            by_res: dict[str, list[str]] = {}
            for it in s.interactions:
                by_res.setdefault(it["residue"], []).append(it["type"])
            for r in s.residues:
                w.writerow([s.name, s.kind, r["chain"], r["resname"], r["resseq"], r["icode"],
                            r["class"], f"{r['min_dist']:.2f}",
                            ";".join(sorted(set(by_res.get(r["key"], [])))) ])
    return path


_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<script src="__3DMOL__"></script>
<style>
:root{--bg:#f6f7f9;--panel:#fff;--text:#1d2330;--muted:#5d6676;--line:#e1e4ea;--accent:#2f6fde;
--hp:#d9a441;--ar:#a36ad9;--po:#3fae8c;--pos:#3d7be0;--neg:#e0533d;--ot:#9aa3b0}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#14171c;--panel:#1c2027;
--text:#e6e9ef;--muted:#9aa3b2;--line:#2c323c;--accent:#6d9cf0}}
:root[data-theme=dark]{--bg:#14171c;--panel:#1c2027;--text:#e6e9ef;--muted:#9aa3b2;--line:#2c323c;--accent:#6d9cf0}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif;
height:100vh;display:grid;grid-template-columns:1fr 400px;grid-template-rows:auto 1fr}
header{grid-column:1/-1;padding:10px 16px;border-bottom:1px solid var(--line);background:var(--panel);
display:flex;gap:16px;align-items:center;flex-wrap:wrap}
header h1{font-size:17px;margin:0}
header .sub{color:var(--muted);font-size:12.5px;max-width:60ch;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
#viewer{position:relative;min-height:300px}
#hover{position:absolute;left:10px;bottom:10px;background:rgba(0,0,0,.7);color:#fff;padding:3px 8px;border-radius:4px;
font-size:12px;pointer-events:none;display:none;z-index:5}
aside{background:var(--panel);border-left:1px solid var(--line);overflow-y:auto;padding:12px 14px 40px}
h2{font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);margin:16px 0 8px}
.controls{display:grid;grid-template-columns:auto 1fr;gap:6px 10px;align-items:center}
.controls label{color:var(--muted);font-size:12.5px}
select,button{font:inherit;font-size:13px;padding:4px 8px;border:1px solid var(--line);border-radius:6px;
background:var(--panel);color:var(--text);cursor:pointer}
button:hover{border-color:var(--accent)}
.row{display:flex;gap:6px;flex-wrap:wrap;margin-top:8px}
.chk{display:flex;gap:6px;align-items:center;font-size:13px}
.site{border:1px solid var(--line);border-radius:8px;padding:10px;margin-bottom:10px;border-left:5px solid var(--c)}
.site.active{box-shadow:0 0 0 2px color-mix(in srgb,var(--c) 45%,transparent)}
.site-head{display:flex;align-items:center;gap:8px}
.site-head b{flex:1}
.kind{font-size:11px;color:var(--muted);border:1px solid var(--line);border-radius:10px;padding:0 6px}
.desc{color:var(--muted);font-size:12.5px;margin:4px 0}
.stats{display:flex;flex-wrap:wrap;gap:4px 14px;font-size:12.5px;margin:6px 0}
.stats span{white-space:nowrap}
.stats span b{font-variant-numeric:tabular-nums}
.bar{display:flex;height:8px;border-radius:4px;overflow:hidden;margin:6px 0 2px}
.legend{display:flex;flex-wrap:wrap;gap:4px 10px;font-size:11px;color:var(--muted)}
.legend i{display:inline-block;width:8px;height:8px;border-radius:2px;margin-right:3px}
details{margin-top:6px}
summary{cursor:pointer;font-size:12.5px;color:var(--accent)}
table{width:100%;border-collapse:collapse;font-size:12px;margin-top:4px}
th,td{text-align:left;padding:3px 4px;border-bottom:1px solid var(--line);white-space:nowrap}
th{color:var(--muted);font-weight:500}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
tr.click{cursor:pointer}tr.click:hover{background:color-mix(in srgb,var(--accent) 10%,transparent)}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:4px}
.note{font-size:11.5px;color:var(--muted);margin-top:14px}
@media (max-width:820px){body{grid-template-columns:1fr;grid-template-rows:auto 55vh auto;height:auto}
aside{border-left:0;border-top:1px solid var(--line)}}
</style>
</head>
<body>
<header><h1 id="ttl"></h1><div class="sub" id="sub"></div></header>
<div id="viewer"><div id="hover"></div></div>
<aside>
  <h2>Display</h2>
  <div class="controls">
    <label for="pstyle">Protein</label>
    <select id="pstyle"><option value="cartoon">Cartoon</option><option value="cartoon+surface">Cartoon + surface</option>
      <option value="surface">Surface</option><option value="line">Lines</option></select>
    <label for="pocketStyle">Pocket volume</label>
    <select id="pocketStyle"><option value="spheres">Probe points</option><option value="blob">Surface blob</option>
      <option value="none">Hidden</option></select>
    <label for="surfColor">Site surface</label>
    <select id="surfColor"><option value="none">Off</option><option value="site">Site color</option>
      <option value="class">Residue property</option></select>
  </div>
  <div class="row">
    <span class="chk"><input type="checkbox" id="labels" checked><label for="labels">Labels (focused site)</label></span>
    <span class="chk"><input type="checkbox" id="inter" checked><label for="inter">Interactions</label></span>
    <span class="chk"><input type="checkbox" id="others"><label for="others">Other HETATMs</label></span>
  </div>
  <div class="row">
    <button id="reset">Reset view</button><button id="bg">Dark background</button><button id="png">Save PNG</button>
  </div>
  <h2>Selected binding sites</h2>
  <div id="sites"></div>
  <h2>All detected pockets</h2>
  <div class="desc">Ranked by volume × buriedness². Tick to overlay.</div>
  <table id="pockets"><thead><tr><th></th><th>#</th><th class="num">Vol Å³</th><th class="num">Buried</th>
    <th class="num">Res</th><th>Overlaps</th></tr></thead><tbody></tbody></table>
  <div class="note" id="params"></div>
</aside>
<script>
const DATA = __DATA__;
const CLASS_COLORS = {hydrophobic:'--hp',aromatic:'--ar',polar:'--po',positive:'--pos',negative:'--neg',other:'--ot'};
const INTER_COLORS = {'hydrogen bond':'#2f7fe0','salt bridge':'#e0453d','hydrophobic':'#9a9a9a','metal':'#9b59d0'};
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));

$('ttl').textContent = DATA.name + ' — binding sites';
$('sub').textContent = DATA.title; $('sub').title = DATA.title;
const p = DATA.params;
$('params').textContent = `Pocket detection: grid ${p.spacing} Å, probe ${p.probe} Å, buriedness ≥ ${p.min_buriedness} ` +
  `(${p.n_dirs} rays, ${p.ray_length} Å), min volume ${p.min_volume} Å³. Interaction lines are distance-based candidates ` +
  `(no protonation or angle checks).`;

let darkBg = false, activeSite = null;
const viewer = $3Dmol.createViewer('viewer', {backgroundColor: 'white', antialias: true});
const protein = viewer.addModel(DATA.pdb, 'pdb');
function pointsModel(points){
  const lines = points.map((q, i) => 'HETATM' + String(i % 100000).padStart(5) + '  H   PKT P   1    ' +
    q[0].toFixed(3).padStart(8) + q[1].toFixed(3).padStart(8) + q[2].toFixed(3).padStart(8) + '  1.00  0.00           H');
  const m = viewer.addModel(lines.join('\n') + '\nEND\n', 'pdb', {keepH: true});
  m.setStyle({}, {}); return m;
}
const sites = DATA.sites.map(s => ({...s, visible: true, model: s.points.length ? pointsModel(s.points) : null,
  serials: s.residues.flatMap(r => r.atoms)}));
const pockets = DATA.pockets.map((pk, i) => ({...pk, visible: false, model: pointsModel(pk.points),
  color: `hsl(${(i * 47 + 200) % 360} 55% 55%)`}));
const elemMap = c => ({...$3Dmol.elementColors.Jmol, C: c});
const hexToInt = h => parseInt(h.replace('#',''), 16);

function render(){
  viewer.removeAllShapes(); viewer.removeAllLabels(); viewer.removeAllSurfaces();
  const ps = $('pstyle').value, labels = $('labels').checked;
  protein.setStyle({}, {});
  if (ps.includes('cartoon')) protein.setStyle({hetflag:false}, {cartoon:{color: darkBg ? '#6f7784' : '#c9ced6', opacity: 0.7}});
  if (ps === 'line') protein.setStyle({hetflag:false}, {line:{colorscheme:'grayCarbon'}});
  if (ps.includes('surface')) viewer.addSurface($3Dmol.SurfaceType.MS, {opacity: ps === 'surface' ? 0.85 : 0.35,
      color: darkBg ? '#596270' : '#e8ebf0'}, {model: protein, hetflag: false});
  if ($('others').checked) protein.setStyle({hetflag:true, not:{resn:['HOH','WAT']}}, {stick:{radius:0.15, colorscheme:'grayCarbon'}}, true);
  const pstyle = $('pocketStyle').value;
  const drawPoints = (m, color) => {
    if (!m) return;
    if (pstyle === 'spheres') m.setStyle({}, {sphere:{radius:0.3, color, opacity:0.85}});
    else { m.setStyle({}, {}); if (pstyle === 'blob') viewer.addSurface($3Dmol.SurfaceType.VDW, {color, opacity:0.55}, {model: m}); }
  };
  pockets.forEach(pk => { pk.model.setStyle({}, {}); if (pk.visible) drawPoints(pk.model, pk.color); });
  for (const s of sites){
    if (s.model) s.model.setStyle({}, {});
    if (!s.visible) continue;
    drawPoints(s.model, s.color);
    if (s.serials.length){
      protein.setStyle({serial: s.serials}, {stick:{radius:0.17, colorscheme:{prop:'elem', map: elemMap(hexToInt(s.color))}}}, true);
      if (ps.includes('cartoon')) protein.setStyle({serial: s.serials}, {cartoon:{color: s.color}}, true);
    }
    if (s.ligand_serials.length)
      protein.setStyle({serial: s.ligand_serials}, {stick:{radius:0.3, colorscheme:{prop:'elem', map: elemMap(0x00c853)}},
        sphere:{scale:0.3, colorscheme:{prop:'elem', map: elemMap(0x00c853)}}});
    const sc = $('surfColor').value;
    if (sc !== 'none' && s.serials.length){
      const colorfunc = sc === 'class' ? (a => { const r = s.residues.find(r => r.atoms.includes(a.serial));
        return css(CLASS_COLORS[r ? r.class : 'other']); }) : null;
      viewer.addSurface($3Dmol.SurfaceType.MS, sc === 'class' ? {opacity:0.75, colorfunc} : {opacity:0.6, color: s.color},
        {serial: s.serials}, {serial: s.serials});
    }
    if (labels && s === activeSite) for (const r of s.residues){
      const a = protein.selectedAtoms({serial: r.atoms}).find(a => a.atom === 'CA') || protein.selectedAtoms({serial: r.atoms})[0];
      if (a) viewer.addLabel(`${r.resname}${r.resseq}${r.icode}`, {position:{x:a.x, y:a.y, z:a.z}, fontSize: 11,
        fontColor: darkBg ? 'white' : 'black', backgroundColor: darkBg ? 'black' : 'white', backgroundOpacity: 0.55,
        borderThickness: 0, inFront: true, showBackground: true});
    }
    if ($('inter').checked) for (const it of s.interactions){
      viewer.addCylinder({start:{x:it.a[0],y:it.a[1],z:it.a[2]}, end:{x:it.b[0],y:it.b[1],z:it.b[2]}, radius:0.06,
        dashed:true, dashLength:0.25, gapLength:0.18, color: INTER_COLORS[it.type], fromCap:1, toCap:1});
    }
  }
  viewer.render();
}

function focusSite(s){
  activeSite = s; render();
  document.querySelectorAll('.site').forEach((d, i) => d.classList.toggle('active', sites[i] === s));
  if (s.ligand_serials.length) viewer.zoomTo({serial: s.ligand_serials.concat(s.serials)});
  else if (s.serials.length) viewer.zoomTo({serial: s.serials});
  else if (s.model) viewer.zoomTo({model: s.model});
  viewer.zoom(0.85); viewer.render();
}
function focusResidue(r){
  viewer.zoomTo({serial: r.atoms}, 500);
}
function composition(s){
  const counts = {}; s.residues.forEach(r => counts[r.class] = (counts[r.class] || 0) + 1);
  const n = s.residues.length || 1;
  const order = ['hydrophobic','aromatic','polar','positive','negative','other'].filter(k => counts[k]);
  return `<div class="bar" role="img" aria-label="residue composition">${order.map(k =>
    `<span style="width:${100*counts[k]/n}%;background:var(${CLASS_COLORS[k]})" title="${k}: ${counts[k]}"></span>`).join('')}</div>
    <div class="legend">${order.map(k => `<span><i style="background:var(${CLASS_COLORS[k]})"></i>${k} ${counts[k]}</span>`).join('')}</div>`;
}

function buildSites(){
  const host = $('sites');
  if (!sites.length){ host.innerHTML = '<div class="desc">No sites selected.</div>'; return; }
  host.innerHTML = '';
  sites.forEach((s, si) => {
    const div = document.createElement('div'); div.className = 'site'; div.style.setProperty('--c', s.color);
    const byRes = {}; s.interactions.forEach(it => (byRes[it.residue] = byRes[it.residue] || new Set()).add(it.type));
    div.innerHTML = `<div class="site-head"><input type="checkbox" checked aria-label="show site"><b>${esc(s.name)}</b>
      <span class="kind">${s.kind}</span><button>Focus</button></div>
      <div class="desc">${esc(s.description)}</div>
      <div class="stats"><span><b>${s.residues.length}</b> residues</span>
        ${s.volume != null ? `<span><b>${Math.round(s.volume)}</b> Å³ probe volume</span>` : ''}
        ${s.interactions.length ? `<span><b>${s.interactions.length}</b> contacts</span>` : ''}
        ${s.overlapping_pockets.length ? `<span>pocket ${s.overlapping_pockets.map(i => '#' + i).join(', ')}</span>` : ''}</div>
      ${composition(s)}
      <details ${si === 0 ? 'open' : ''}><summary>Residues</summary><table><thead><tr><th>Residue</th><th>Chain</th>
        <th>Class</th><th class="num">Min d Å</th><th>Contacts</th></tr></thead><tbody>${s.residues.map((r, ri) =>
        `<tr class="click" data-r="${ri}"><td>${r.resname}${r.resseq}${r.icode}</td><td>${r.chain || '–'}</td>
         <td><span class="dot" style="background:var(${CLASS_COLORS[r.class]})"></span>${r.class}</td>
         <td class="num">${r.min_dist.toFixed(1)}</td><td>${[...(byRes[r.key] || [])].join(', ')}</td></tr>`).join('')}
      </tbody></table></details>
      ${s.interactions.length ? `<details><summary>Interactions</summary><table><thead><tr><th>Type</th><th>Ligand</th>
        <th>Residue</th><th class="num">d Å</th></tr></thead><tbody>${s.interactions.map(it =>
        `<tr><td><span class="dot" style="background:${INTER_COLORS[it.type]}"></span>${it.type}</td><td>${esc(it.ligand_atom)}</td>
         <td>${esc(it.residue)} ${esc(it.residue_atom)}</td><td class="num">${it.distance.toFixed(2)}</td></tr>`).join('')}
        </tbody></table></details>` : ''}`;
    div.querySelector('input').onchange = e => { s.visible = e.target.checked; render(); };
    div.querySelector('button').onclick = () => focusSite(s);
    div.querySelectorAll('tr.click').forEach(tr => tr.onclick = () => focusResidue(s.residues[+tr.dataset.r]));
    host.appendChild(div);
  });
}
function buildPockets(){
  const ov = {}; sites.forEach(s => s.overlapping_pockets.forEach(i => (ov[i] = ov[i] || []).push(s.name)));
  $('pockets').querySelector('tbody').innerHTML = pockets.map((pk, i) =>
    `<tr><td><input type="checkbox" data-i="${i}" aria-label="show pocket ${pk.id}"></td>
     <td><span class="dot" style="background:${pk.color}"></span>${pk.id}</td><td class="num">${Math.round(pk.volume)}</td>
     <td class="num">${pk.buriedness.toFixed(2)}</td><td class="num">${pk.n_residues}</td>
     <td>${esc((ov[pk.id] || []).join(', '))}</td></tr>`).join('') ||
    '<tr><td colspan="6" class="desc">No pockets passed the thresholds.</td></tr>';
  $('pockets').querySelectorAll('input').forEach(cb => cb.onchange = e => {
    const pk = pockets[+e.target.dataset.i]; pk.visible = e.target.checked; render();
    if (pk.visible) viewer.zoomTo({model: pk.model}, 500);
  });
}

viewer.setHoverable({model: protein}, true, (a) => {
  const h = $('hover'); h.style.display = 'block';
  h.textContent = `${a.resn} ${a.chain ? a.chain + ':' : ''}${a.resi}  ${a.atom}  (${a.elem})`;
}, () => { $('hover').style.display = 'none'; });

['pstyle','pocketStyle','surfColor','labels','inter','others'].forEach(id => $(id).onchange = render);
$('reset').onclick = () => { activeSite = null; document.querySelectorAll('.site').forEach(d => d.classList.remove('active'));
  render(); viewer.zoomTo({model: protein}, 600); };
$('bg').onclick = () => { darkBg = !darkBg; viewer.setBackgroundColor(darkBg ? '#111418' : 'white');
  $('bg').textContent = darkBg ? 'Light background' : 'Dark background'; render(); };
$('png').onclick = () => { const a = document.createElement('a'); a.href = viewer.pngURI();
  a.download = DATA.name + '_sites.png'; a.click(); };

buildSites(); buildPockets(); render();
viewer.zoomTo({model: protein});
if (sites.length) focusSite(sites[0]);
window.addEventListener('resize', () => viewer.resize());
</script>
</body>
</html>
"""
