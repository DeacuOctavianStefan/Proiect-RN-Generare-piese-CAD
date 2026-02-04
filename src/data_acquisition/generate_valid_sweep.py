
"""
generate_valid_sweep.py

Generate constraint-valid JSONL records for SolidWorks part templates (shaft, sleeve, flange).

- Uses the *_template_target_updated.jsonl "base record" as a schema template.
- Produces JSONL where each line is a record with updated template_inputs.
- Designed to work with your dimension-driven generator script.

Assumptions / conventions (edit if needed):
- Units in JSON are millimeters for lengths, degrees for angles.
- For flanges, BOLT_PCD is treated as *bolt circle radius* (not diameter), because your
  flange_template_target_updated.jsonl uses BOLT_PCD=7.5 for OD=20 (i.e., PCD=15).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import uuid
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# -----------------------------
# Helpers
# -----------------------------

def load_base_record(path: Path) -> Dict[str, Any]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"Base JSONL file is empty: {path}")
    return json.loads(lines[0])

def write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def new_id() -> str:
    return uuid.uuid4().hex[:8]

def mm(x: float) -> float:
    # keep as mm in JSON (just a semantic helper)
    return float(x)

def deg(x: float) -> float:
    return float(x)

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

# -----------------------------
# Valid generators
# -----------------------------

def gen_shaft_record(base: Dict[str, Any], rng: random.Random, idx: int) -> Dict[str, Any]:
    # Tunables
    wall_min = 2.0          # mm
    chamfer_l_max = 6.0     # mm
    chamfer_l_min = 0.2     # mm

    # Sample core dims
    shaft_od = rng.randrange(10, 121, 5)  # 10..120 step 5
    # Bore must fit with wall thickness
    bore_max = max(0.0, shaft_od - 2.0 * wall_min)
    bore_d = rng.uniform(0.3 * bore_max, 0.9 * bore_max) if bore_max >= 5 else max(1.0, bore_max * 0.8)
    bore_d = clamp(bore_d, 1.0, max(1.0, bore_max))

    # Length: tie to OD
    shaft_l = rng.uniform(max(20.0, 2.0 * shaft_od), 10.0 * shaft_od)
    shaft_l = clamp(shaft_l, 20.0, 600.0)

    # Chamfer
    chamfer_l = rng.uniform(chamfer_l_min, min(chamfer_l_max, shaft_od * 0.12))
    chamfer_ang = rng.choice([30.0, 45.0, 60.0]) if rng.random() < 0.7 else rng.uniform(20.0, 70.0)

    rec = dict(base)
    rec["id"] = new_id()
    rec["file_name"] = f"shaft_{rec['id']}.SLDPRT"
    rec["prompt"] = f"Generate a bored shaft with OD {shaft_od:.0f}mm, length {shaft_l:.0f}mm, bore {bore_d:.0f}mm."
    rec["template_inputs"] = {
        "SHAFT_OD": mm(shaft_od),
        "SHAFT_L": mm(shaft_l),
        "BORE_D": mm(bore_d),
        # your rename: CHAMFER_L (SolidWorks Link Values name)
        "CHAMFER_L": mm(chamfer_l),
        "CHAMFER_ANG": deg(chamfer_ang),
    }
    return rec

def gen_sleeve_record(base: Dict[str, Any], rng: random.Random, idx: int) -> Dict[str, Any]:
    wall_min = 2.0
    chamfer_l_max = 4.0
    chamfer_l_min = 0.2

    sleeve_od = rng.randrange(10, 121, 5)
    bore_max = max(0.0, sleeve_od - 2.0 * wall_min)
    bore_d = rng.uniform(0.4 * bore_max, 0.95 * bore_max) if bore_max >= 5 else max(1.0, bore_max * 0.8)
    bore_d = clamp(bore_d, 1.0, max(1.0, bore_max))

    # Sleeve length typically modest
    sleeve_l = rng.uniform(max(5.0, 0.3 * sleeve_od), 3.0 * sleeve_od)
    sleeve_l = clamp(sleeve_l, 5.0, 400.0)

    chamfer_l = rng.uniform(chamfer_l_min, min(chamfer_l_max, sleeve_od * 0.10))
    chamfer_ang = rng.choice([30.0, 45.0, 60.0]) if rng.random() < 0.7 else rng.uniform(20.0, 70.0)

    rec = dict(base)
    rec["id"] = new_id()
    rec["file_name"] = f"bearing_sleeve_{rec['id']}.SLDPRT"
    rec["prompt"] = f"Generate a sleeve bushing with OD {sleeve_od:.0f}mm, length {sleeve_l:.0f}mm, bore {bore_d:.0f}mm."
    rec["template_inputs"] = {
        "SLEEVE_OD": mm(sleeve_od),
        "SLEEVE_L": mm(sleeve_l),
        "BORE_D": mm(bore_d),
        "CHAMFER_L": mm(chamfer_l),
        "CHAMFER_ANG": deg(chamfer_ang),
    }
    return rec

def gen_flange_record(base: Dict[str, Any], rng: random.Random, idx: int) -> Dict[str, Any]:
    """
    Flange validity constraints (all mm, deg):
      - BORE_D < FLANGE_OD
      - Bolt circle radius (BOLT_PCD) must satisfy:
          inner_clear:  r - bolt_d/2 > bore_d/2 + clearance
          outer_clear:  r + bolt_d/2 < flange_od/2 - edge_margin
          hole_spacing: chord length between holes > bolt_d + spacing_min
      - Bolt depth <= flange thickness
    Convention:
      - BOLT_PCD is *radius* (mm).
    """
    clearance = 1.0      # mm from bore edge
    edge_margin = 1.5    # mm from OD edge
    spacing_min = 1.0    # mm minimum material between holes along chord
    wall_min = 2.5       # mm minimum radial wall between bore and OD (not counting bolts)

    # 1) Bore
    bore_d = rng.randrange(10, 101, 5)  # 10..100
    # 2) Flange OD must exceed bore with wall
    flange_od_min = bore_d + 2.0 * wall_min + 10.0
    flange_od = rng.randrange(int(math.ceil(flange_od_min / 5.0)) * 5, 201, 5)  # up to 200mm
    flange_od = clamp(flange_od, flange_od_min, 200.0)

    # 3) Thickness
    flange_thk = rng.randrange(4, 31, 1)
    flange_thk = clamp(flange_thk, 3.0, max(3.0, 0.4 * flange_od))

    # 4) Bolt count
    bolt_count = rng.choice([3, 4, 5, 6, 8])

    # 5) Bolt diameter (keep reasonable vs flange)
    max_bolt_d = max(2.5, 0.18 * (flange_od - bore_d))
    bolt_d = rng.uniform(3.0, max_bolt_d)
    bolt_d = clamp(bolt_d, 3.0, 20.0)

    # 6) Choose bolt circle radius r with constraints
    bore_r = bore_d / 2.0
    flange_r = flange_od / 2.0

    # inner / outer clearance bounds
    r_min_inner = bore_r + clearance + bolt_d / 2.0
    r_max_outer = flange_r - edge_margin - bolt_d / 2.0

    # spacing constraint: chord = 2*r*sin(pi/n) > bolt_d + spacing_min
    if bolt_count >= 3:
        sin_term = math.sin(math.pi / bolt_count)
        r_min_spacing = (bolt_d + spacing_min) / (2.0 * sin_term) if sin_term > 1e-6 else r_min_inner
    else:
        r_min_spacing = r_min_inner

    r_min = max(r_min_inner, r_min_spacing)
    r_max = r_max_outer

    # If infeasible, relax bolt_d down a bit or widen flange_od (retry loop)
    for _ in range(20):
        if r_min < r_max:
            break
        # relax: reduce bolt_d and recompute
        bolt_d = max(2.5, bolt_d * 0.85)
        r_min_inner = bore_r + clearance + bolt_d / 2.0
        sin_term = math.sin(math.pi / bolt_count)
        r_min_spacing = (bolt_d + spacing_min) / (2.0 * sin_term) if sin_term > 1e-6 else r_min_inner
        r_min = max(r_min_inner, r_min_spacing)
        r_max = flange_r - edge_margin - bolt_d / 2.0

    if not (r_min < r_max):
        # Last resort: bump flange_od and recompute once
        flange_od = clamp(flange_od + 20.0, flange_od_min, 220.0)
        flange_r = flange_od / 2.0
        r_max = flange_r - edge_margin - bolt_d / 2.0

    if not (r_min < r_max):
        # Still infeasible: mark with conservative values
        r_min = bore_r + clearance + bolt_d / 2.0
        r_max = r_min + 1.0

    bolt_pcd_r = rng.uniform(r_min, r_max)
    bolt_pcd_r = clamp(bolt_pcd_r, r_min, r_max)

    # 7) Bolt depth (cut depth)
    bolt_depth = rng.uniform(0.6 * flange_thk, 1.0 * flange_thk)
    bolt_depth = clamp(bolt_depth, 1.0, flange_thk)

    rec = dict(base)
    rec["id"] = new_id()
    rec["file_name"] = f"bearing_flanged_{rec['id']}.SLDPRT"
    # Note: We deliberately describe PCD as diameter in the prompt (human-friendly), while storing radius in BOLT_PCD.
    rec["prompt"] = (
        f"Generate a flanged bearing with OD {flange_od:.0f}mm, thickness {flange_thk:.0f}mm, "
        f"bore {bore_d:.0f}mm, {bolt_count} bolt holes Ø{bolt_d:.1f}mm on PCD {(2*bolt_pcd_r):.1f}mm."
    )
    rec["template_inputs"] = {
        "FLANGE_OD": mm(flange_od),
        "FLANGE_THK": mm(flange_thk),
        "BORE_D": mm(bore_d),
        "BOLT_D": mm(bolt_d),
        "BOLT_COUNT": int(bolt_count),
        "BOLT_DEPTH": mm(bolt_depth),
        # IMPORTANT: radius convention
        "BOLT_PCD": mm(bolt_pcd_r),
    }
    return rec

# -----------------------------
# Validation (optional)
# -----------------------------

def validate_flange(t: Dict[str, Any]) -> Tuple[bool, str]:
    od = float(t["FLANGE_OD"])
    thk = float(t["FLANGE_THK"])
    bore = float(t["BORE_D"])
    bolt_d = float(t["BOLT_D"])
    n = int(t["BOLT_COUNT"])
    depth = float(t["BOLT_DEPTH"])
    r = float(t["BOLT_PCD"])  # radius

    if bore <= 0 or od <= 0 or thk <= 0:
        return False, "non-positive dims"
    if bore >= od:
        return False, "bore >= od"
    if depth > thk + 1e-6:
        return False, "bolt_depth > flange_thk"
    bore_r = bore / 2.0
    flange_r = od / 2.0
    # clearances
    if r - bolt_d/2.0 <= bore_r + 0.5:
        return False, "bolt circle too close to bore"
    if r + bolt_d/2.0 >= flange_r - 0.5:
        return False, "bolt circle too close to OD"
    # spacing
    chord = 2.0 * r * math.sin(math.pi / max(3, n))
    if chord <= bolt_d + 0.5:
        return False, "bolt holes too close"
    return True, "ok"

# -----------------------------
# CLI
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", type=str, default=".", help="Repo root (used to locate data/raw/*.jsonl base records)")
    ap.add_argument("--out_dir", type=str, default="data/raw/generated_sweep_valid", help="Output directory for generated JSONLs")
    ap.add_argument("--n", type=int, default=2000, help="How many records per family")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--families", type=str, default="all", help="Comma-separated: flange,shaft,sleeve,all")
    ap.add_argument("--validate", type=int, default=1, help="Validate flange constraints and resample if needed")
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    out_dir = (repo / args.out_dir).resolve()
    rng = random.Random(args.seed)

    fams = [s.strip().lower() for s in args.families.split(",")]
    if "all" in fams:
        fams = ["flange", "shaft", "sleeve"]

    # base records
    base_flange = load_base_record(repo / "data/raw/flange_template_target_updated.jsonl")
    base_shaft  = load_base_record(repo / "data/raw/shaft_template_target_updated.jsonl")
    base_sleeve = load_base_record(repo / "data/raw/sleeve_template_target_updated.jsonl")

    generated: Dict[str, List[Dict[str, Any]]] = {}

    if "shaft" in fams:
        recs = [gen_shaft_record(base_shaft, rng, i) for i in range(args.n)]
        generated["shaft"] = recs
        write_jsonl(out_dir / "shaft.jsonl", recs)

    if "sleeve" in fams:
        recs = [gen_sleeve_record(base_sleeve, rng, i) for i in range(args.n)]
        generated["sleeve"] = recs
        write_jsonl(out_dir / "sleeve.jsonl", recs)

    if "flange" in fams:
        recs: List[Dict[str, Any]] = []
        for i in range(args.n):
            # resample until valid (bounded)
            for _ in range(50):
                r = gen_flange_record(base_flange, rng, i)
                if not args.validate:
                    recs.append(r); break
                ok, _msg = validate_flange(r["template_inputs"])
                if ok:
                    recs.append(r); break
            else:
                # If we fail to find a valid one, still append last attempt for debugging
                recs.append(r)
        generated["flange"] = recs
        write_jsonl(out_dir / "flange.jsonl", recs)

    print(f"Wrote JSONLs to: {out_dir}")
    for k, v in generated.items():
        print(f"  {k}: {len(v)} records")

if __name__ == "__main__":
    main()
