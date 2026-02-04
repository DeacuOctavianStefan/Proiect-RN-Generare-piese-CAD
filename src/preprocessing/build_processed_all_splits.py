import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple


FAMILIES = ["shaft", "sleeve", "flange"]
SPLITS = ["train", "validation", "test"]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def canon_inputs(d: Dict[str, Any]) -> str:
    return json.dumps(d, sort_keys=True, ensure_ascii=False)


def load_manifest_ok(manifest_csv: Path) -> List[Dict[str, Any]]:
    ok = []
    if not manifest_csv.exists():
        return ok
    with manifest_csv.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if (row.get("status") or "").strip().upper() == "OK":
                ok.append(row)
    return ok


def load_prompts_index(prompts_jsonl: Path) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, List[Dict[str, Any]]]]:
    by_id: Dict[str, List[Dict[str, Any]]] = {}
    by_inputs: Dict[str, List[Dict[str, Any]]] = {}
    rows = read_jsonl(prompts_jsonl)
    for rec in rows:
        rec_id = str(rec.get("id", "")).strip()
        ti = rec.get("template_inputs") or {}
        key = canon_inputs(ti)
        by_inputs.setdefault(key, []).append(rec)
        if rec_id:
            by_id.setdefault(rec_id, []).append(rec)
    return by_id, by_inputs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", required=True)
    ap.add_argument("--prompts_dir", default="data/raw/generated_prompts")
    ap.add_argument("--sweeps_dir", default="data/raw/generated_sweep_valid")
    ap.add_argument("--parts_root", default="data")  # expects data/<split>/parts/<family>/manifest.csv
    ap.add_argument("--out_dir", default="data/processed")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    prompts_dir = repo / args.prompts_dir
    sweeps_dir = repo / args.sweeps_dir
    parts_root = repo / args.parts_root
    out_dir = repo / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    out_rows: Dict[str, List[Dict[str, Any]]] = {s: [] for s in SPLITS}

    for fam in FAMILIES:
        sweep_path = sweeps_dir / f"{fam}.jsonl"
        prompts_path = prompts_dir / f"{fam}.jsonl"

        if not sweep_path.exists():
            print(f"[skip] missing sweep: {sweep_path}")
            continue
        if not prompts_path.exists():
            print(f"[skip] missing prompts: {prompts_path}")
            continue

        sweep_rows = read_jsonl(sweep_path)
        by_id, by_inputs = load_prompts_index(prompts_path)

        for split in SPLITS:
            manifest = parts_root / split / "parts" / fam / "manifest.csv"
            ok_rows = load_manifest_ok(manifest)
            if not ok_rows:
                print(f"[warn] no OK parts for {fam} split={split}: {manifest}")
                continue

            # Build joinable part list
            parts: List[Tuple[int, str, str, Dict[str, Any]]] = []
            for row in ok_rows:
                try:
                    idx = int(row["index"])
                except Exception:
                    continue
                if idx < 0 or idx >= len(sweep_rows):
                    continue
                sweep_rec = sweep_rows[idx]
                ti = sweep_rec.get("template_inputs") or {}
                rec_id = str(sweep_rec.get("id") or sweep_rec.get("rec_id") or idx)
                out_path = row.get("out_path") or ""
                parts.append((idx, out_path, rec_id, ti))

            rng.shuffle(parts)

            # Expand into prompt rows
            for idx, out_path, rec_id, ti in parts:
                prompts = by_id.get(str(rec_id), [])
                if not prompts:
                    prompts = by_inputs.get(canon_inputs(ti), [])

                if not prompts:
                    out_rows[split].append({
                        "id": str(rec_id),
                        "family": fam,
                        "split": split,
                        "prompt": None,
                        "template_inputs": ti,
                        "part_path": out_path,
                        "index": idx,
                        "variant": None,
                        "note": "NO_PROMPT_MATCH",
                    })
                    continue

                for p in prompts:
                    out_rows[split].append({
                        "id": str(p.get("id", rec_id)),
                        "family": p.get("family", fam),
                        "split": split,
                        "prompt": p.get("prompt"),
                        "template_inputs": p.get("template_inputs", ti),
                        "variant": p.get("variant"),
                        "part_path": out_path,
                        "index": idx,
                        "template": p.get("template"),
                    })

            print(f"[ok] joined {fam} split={split}: parts={len(parts)} -> rows={len([r for r in out_rows[split] if (r.get('family')==fam or r.get('family')==p.get('family',fam))])}")

    # Write
    write_jsonl(out_dir / "train.jsonl", out_rows["train"])
    write_jsonl(out_dir / "validation.jsonl", out_rows["validation"])
    write_jsonl(out_dir / "test.jsonl", out_rows["test"])

    print("[done] wrote:")
    print(f"  - {out_dir / 'train.jsonl'} ({len(out_rows['train'])} rows)")
    print(f"  - {out_dir / 'validation.jsonl'} ({len(out_rows['validation'])} rows)")
    print(f"  - {out_dir / 'test.jsonl'} ({len(out_rows['test'])} rows)")
    print("If you see note=NO_PROMPT_MATCH rows, tell me and we’ll tighten the ID/index joining.")


if __name__ == "__main__":
    main()
