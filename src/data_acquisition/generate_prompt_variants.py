import argparse
import json
import random
import re
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def mm(x: Any) -> str:
    # input values are mm/deg in your JSON sweeps
    if x is None:
        return "?"
    try:
        # keep integers clean
        v = float(x)
        if abs(v - round(v)) < 1e-9:
            return f"{int(round(v))} mm"
        # up to 3 decimals for non-integers
        return f"{v:.3f} mm".rstrip("0").rstrip(".") + " mm"
    except Exception:
        return str(x)


def deg(x: Any) -> str:
    if x is None:
        return "?"
    try:
        v = float(x)
        if abs(v - round(v)) < 1e-9:
            return f"{int(round(v))}°"
        return f"{v:.2f}°".rstrip("0").rstrip(".") + "°"
    except Exception:
        return str(x)


def pick(d: Dict[str, Any], *keys: str) -> Optional[Any]:
    for k in keys:
        if k in d:
            return d[k]
    return None


def normalize_family(family: str) -> str:
    f = (family or "").strip().lower()
    if f in {"shaft"}:
        return "shaft"
    if f in {"sleeve", "bearing_sleeve"}:
        return "bearing_sleeve"
    if f in {"flange", "bearing_flanged"}:
        return "bearing_flanged"
    return f


def fmt_shaft(inputs: Dict[str, Any]) -> Dict[str, str]:
    od = pick(inputs, "SHAFT_OD")
    L = pick(inputs, "SHAFT_L")
    bore = pick(inputs, "BORE_D")
    cham_l = pick(inputs, "CHAMFER_L", "CHAMFER_D")  # you renamed CHAMFER_D -> CHAMFER_L
    cham_a = pick(inputs, "CHAMFER_ANG")
    return {
        "od": mm(od),
        "L": mm(L),
        "bore": mm(bore),
        "cham_l": mm(cham_l),
        "cham_a": deg(cham_a),
    }


def fmt_sleeve(inputs: Dict[str, Any]) -> Dict[str, str]:
    od = pick(inputs, "SLEEVE_OD")
    L = pick(inputs, "SLEEVE_L")
    bore = pick(inputs, "BORE_D")
    cham_l = pick(inputs, "CHAMFER_L", "CHAMFER_D")
    cham_a = pick(inputs, "CHAMFER_ANG")
    return {
        "od": mm(od),
        "L": mm(L),
        "bore": mm(bore),
        "cham_l": mm(cham_l),
        "cham_a": deg(cham_a),
    }


def fmt_flange(inputs: Dict[str, Any], pcd_mode: str) -> Dict[str, str]:
    od = pick(inputs, "FLANGE_OD")
    thk = pick(inputs, "FLANGE_THK")
    bore = pick(inputs, "BORE_D")
    bolt_d = pick(inputs, "BOLT_D")
    bolt_depth = pick(inputs, "BOLT_DEPTH")
    bolt_count = pick(inputs, "BOLT_COUNT")
    pcd = pick(inputs, "BOLT_PCD")

    # prompt wording only; values unchanged
    pcd_label = "bolt circle diameter (PCD)" if pcd_mode == "diameter" else "bolt circle radius"
    return {
        "od": mm(od),
        "thk": mm(thk),
        "bore": mm(bore),
        "bolt_d": mm(bolt_d),
        "bolt_depth": mm(bolt_depth),
        "bolt_count": str(int(float(bolt_count))) if bolt_count is not None else "?",
        "pcd": mm(pcd),
        "pcd_label": pcd_label,
    }


def variants_shaft(f: Dict[str, str]) -> List[Tuple[str, str]]:
    return [
        ("spec_lines",
         f"Shaft\n- Outer diameter: {f['od']}\n- Length: {f['L']}\n- Bore diameter: {f['bore']}\n- Chamfer: {f['cham_l']} at {f['cham_a']}"),
        ("compact_csv",
         f"shaft, OD={f['od']}, L={f['L']}, bore={f['bore']}, chamfer={f['cham_l']}, angle={f['cham_a']}"),
        ("natural_1",
         f"Create a shaft with outer diameter {f['od']} and length {f['L']}. Add a coaxial bore of {f['bore']} and a chamfer of {f['cham_l']} at {f['cham_a']}."),
        ("natural_2",
         f"Model a cylindrical shaft: OD {f['od']}, length {f['L']}, central bore {f['bore']}, chamfer {f['cham_l']} @ {f['cham_a']}."),
        ("instructional",
         f"Generate a solid shaft. Set OD={f['od']}, overall length={f['L']}, drill a through bore of {f['bore']}, then apply chamfer {f['cham_l']} with angle {f['cham_a']}."),
    ]


def variants_sleeve(f: Dict[str, str]) -> List[Tuple[str, str]]:
    return [
        ("spec_lines",
         f"Sleeve bushing\n- Outer diameter: {f['od']}\n- Length: {f['L']}\n- Bore diameter: {f['bore']}\n- Chamfer: {f['cham_l']} at {f['cham_a']}"),
        ("compact_csv",
         f"sleeve, OD={f['od']}, L={f['L']}, bore={f['bore']}, chamfer={f['cham_l']}, angle={f['cham_a']}"),
        ("natural_1",
         f"Create a sleeve bushing with OD {f['od']} and length {f['L']}. The internal bore should be {f['bore']}. Add a chamfer of {f['cham_l']} at {f['cham_a']}."),
        ("natural_2",
         f"Model a bushing: outer Ø {f['od']}, length {f['L']}, inner Ø {f['bore']}, chamfer {f['cham_l']} @ {f['cham_a']}."),
        ("instructional",
         f"Generate a sleeve. Set outer diameter={f['od']}, length={f['L']}, cut a concentric bore {f['bore']}, then apply chamfer {f['cham_l']} with angle {f['cham_a']}."),
    ]


def variants_flange(f: Dict[str, str]) -> List[Tuple[str, str]]:
    return [
        ("spec_lines",
         "Flanged bearing (simple)\n"
         f"- Flange OD: {f['od']}\n"
         f"- Thickness: {f['thk']}\n"
         f"- Bore diameter: {f['bore']}\n"
         f"- Bolt hole diameter: {f['bolt_d']}\n"
         f"- Bolt hole depth: {f['bolt_depth']}\n"
         f"- Bolt count: {f['bolt_count']}\n"
         f"- {f['pcd_label']}: {f['pcd']}"),
        ("compact_csv",
         f"flange, OD={f['od']}, thk={f['thk']}, bore={f['bore']}, bolt_d={f['bolt_d']}, bolt_depth={f['bolt_depth']}, bolts={f['bolt_count']}, pcd={f['pcd']}"),
        ("natural_1",
         f"Create a simple flange with outer diameter {f['od']} and thickness {f['thk']}. "
         f"Add a central bore of {f['bore']}. "
         f"Add {f['bolt_count']} bolt holes of diameter {f['bolt_d']} and depth {f['bolt_depth']} "
         f"on the {f['pcd_label']} of {f['pcd']}."),
        ("instructional",
         f"Model a flanged part: set OD={f['od']}, thickness={f['thk']}, bore={f['bore']}. "
         f"Create a circular bolt pattern with {f['bolt_count']} holes. Hole Ø={f['bolt_d']}, depth={f['bolt_depth']}, "
         f"{f['pcd_label']}={f['pcd']}."),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", required=True)
    ap.add_argument("--in_dir", default="data/raw/generated_sweep_valid", help="Folder containing sweep JSONLs")
    ap.add_argument("--out_dir", default="data/raw/generated_prompts", help="Output folder for prompt JSONLs")
    ap.add_argument("--families", default="all", help="shaft|sleeve|flange|all")
    ap.add_argument("--variants_per_record", type=int, default=4, help="How many prompt variants to keep per record")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pcd_mode", choices=["radius", "diameter"], default="radius",
                    help="How to describe BOLT_PCD in prompts. Values are not changed; only wording.")
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    in_dir = repo_root / args.in_dir
    out_dir = repo_root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    fams = args.families.strip().lower()
    if fams == "all":
        selected = ["shaft", "sleeve", "flange"]
    else:
        selected = [x.strip() for x in re.split(r"[,\s]+", fams) if x.strip()]

    rng = random.Random(args.seed)

    family_to_file = {
        "shaft": "shaft.jsonl",
        "sleeve": "sleeve.jsonl",
        "flange": "flange.jsonl",
    }

    for fam in selected:
        in_path = in_dir / family_to_file[fam]
        if not in_path.exists():
            print(f"[skip] missing input: {in_path}")
            continue

        rows_out: List[Dict[str, Any]] = []

        for rec in read_jsonl(in_path):
            rec_id = rec.get("id") or rec.get("rec_id") or str(uuid.uuid4())
            inputs = rec.get("template_inputs") or rec.get("inputs") or {}

            if fam == "shaft":
                f = fmt_shaft(inputs)
                cand = variants_shaft(f)
                # randomly choose subset
                rng.shuffle(cand)
                cand = cand[: max(1, min(args.variants_per_record, len(cand)))]
            elif fam == "sleeve":
                f = fmt_sleeve(inputs)
                cand = variants_sleeve(f)
                rng.shuffle(cand)
                cand = cand[: max(1, min(args.variants_per_record, len(cand)))]
            else:
                f = fmt_flange(inputs, args.pcd_mode)
                cand = variants_flange(f)
                rng.shuffle(cand)
                cand = cand[: max(1, min(args.variants_per_record, len(cand)))]

            family_norm = normalize_family(rec.get("family") or fam)

            for variant_name, prompt in cand:
                rows_out.append({
                    "id": rec_id,
                    "family": family_norm,
                    "variant": variant_name,
                    "prompt": prompt,
                    "template_inputs": inputs,
                    # keep template path if present (useful later)
                    "template": rec.get("template"),
                })

        out_path = out_dir / f"{fam}.jsonl"
        write_jsonl(out_path, rows_out)
        print(f"[ok] wrote {len(rows_out)} prompts -> {out_path}")


if __name__ == "__main__":
    main()
