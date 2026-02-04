import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple


FAMILY_FOLDERS = ["shaft", "sleeve", "flange"]


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
    """Canonical string for matching template_inputs regardless of key order."""
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
    """
    Returns:
      by_id: id -> [prompt records...]
      by_inputs: canon(template_inputs) -> [prompt records...]
    """
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

    # desired per-family counts (per split)
    ap.add_argument("--train_n", type=int, default=1600)
    ap.add_argument("--val_n", type=int, default=200)
    ap.add_argument("--test_n", type=int, default=200)

    # where the OK parts currently live (you generated train only so far)
    ap.add_argument("--source_split", choices=["train", "validation", "test"], default="train",
                    help="Which split folder currently contains the generated parts/manifests to sample from.")
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    prompts_dir = repo / args.prompts_dir
    sweeps_dir = repo / args.sweeps_dir
    parts_root = repo / args.parts_root
    out_dir = repo / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    all_train: List[Dict[str, Any]] = []
    all_val: List[Dict[str, Any]] = []
    all_test: List[Dict[str, Any]] = []

    for fam_folder in FAMILY_FOLDERS:
        manifest = parts_root / args.source_split / "parts" / fam_folder / "manifest.csv"
        ok_rows = load_manifest_ok(manifest)
        if not ok_rows:
            print(f"[skip] no OK rows found for {fam_folder}: {manifest}")
            continue

        # Load sweeps so we can match by index (line number) -> template_inputs
        sweep_path = sweeps_dir / f"{fam_folder}.jsonl"
        if not sweep_path.exists():
            print(f"[skip] missing sweep file for {fam_folder}: {sweep_path}")
            continue
        sweep_rows = read_jsonl(sweep_path)

        # Load prompt variants
        prompts_path = prompts_dir / f"{fam_folder}.jsonl"
        if not prompts_path.exists():
            print(f"[skip] missing prompts file for {fam_folder}: {prompts_path}")
            continue
        by_id, by_inputs = load_prompts_index(prompts_path)

        # Build list of parts (by index) we can join to prompts
        parts: List[Tuple[int, str, str, Dict[str, Any]]] = []
        # tuple: (index, out_path, rec_id, template_inputs)
        for row in ok_rows:
            try:
                idx = int(row["index"])
            except Exception:
                continue
            if idx < 0 or idx >= len(sweep_rows):
                continue

            sweep_rec = sweep_rows[idx]
            ti = sweep_rec.get("template_inputs") or {}
            rec_id = str(sweep_rec.get("id") or sweep_rec.get("rec_id") or idx)  # fallback to idx
            out_path = row.get("out_path") or ""
            parts.append((idx, out_path, rec_id, ti))

        if not parts:
            print(f"[skip] no joinable parts for {fam_folder}")
            continue

        rng.shuffle(parts)

        need_train = args.train_n
        need_val = args.val_n
        need_test = args.test_n
        need_total = need_train + need_val + need_test

        if len(parts) < need_total:
            print(f"[warn] {fam_folder}: only {len(parts)} OK parts available, "
                  f"but requested {need_total}. Will fill train first, then val, then test.")

        train_parts = parts[:need_train]
        val_parts = parts[need_train:need_train + need_val]
        test_parts = parts[need_train + need_val:need_train + need_val + need_test]

        def expand_with_prompts(part_list: List[Tuple[int, str, str, Dict[str, Any]]], split_name: str) -> List[Dict[str, Any]]:
            out: List[Dict[str, Any]] = []
            for idx, out_path, rec_id, ti in part_list:
                # Match prompt variants:
                # 1) by sweep record id
                prompts = by_id.get(str(rec_id), [])
                # 2) fallback by template_inputs canonical match
                if not prompts:
                    prompts = by_inputs.get(canon_inputs(ti), [])

                if not prompts:
                    # keep at least one record (no prompt) so you notice the mismatch
                    out.append({
                        "id": str(rec_id),
                        "family": fam_folder,
                        "split": split_name,
                        "prompt": None,
                        "template_inputs": ti,
                        "part_path": out_path,
                        "index": idx,
                        "variant": None,
                        "note": "NO_PROMPT_MATCH",
                    })
                    continue

                for p in prompts:
                    out.append({
                        "id": str(p.get("id", rec_id)),
                        "family": p.get("family", fam_folder),
                        "split": split_name,
                        "prompt": p.get("prompt"),
                        "template_inputs": p.get("template_inputs", ti),
                        "variant": p.get("variant"),
                        "part_path": out_path,
                        "index": idx,
                        "template": p.get("template"),
                    })
            return out

        train_rows = expand_with_prompts(train_parts, "train")
        val_rows = expand_with_prompts(val_parts, "validation")
        test_rows = expand_with_prompts(test_parts, "test")

        all_train.extend(train_rows)
        all_val.extend(val_rows)
        all_test.extend(test_rows)

        print(f"[ok] {fam_folder}: parts OK={len(parts)} | "
              f"train_parts={len(train_parts)} val_parts={len(val_parts)} test_parts={len(test_parts)} | "
              f"train_rows={len(train_rows)} val_rows={len(val_rows)} test_rows={len(test_rows)}")

    write_jsonl(out_dir / "train.jsonl", all_train)
    write_jsonl(out_dir / "validation.jsonl", all_val)
    write_jsonl(out_dir / "test.jsonl", all_test)

    print(f"[done] wrote:")
    print(f"  - {out_dir / 'train.jsonl'}  ({len(all_train)} rows)")
    print(f"  - {out_dir / 'validation.jsonl'}  ({len(all_val)} rows)")
    print(f"  - {out_dir / 'test.jsonl'}  ({len(all_test)} rows)")
    print("\nNotes:")
    print("- Splitting is by PART (index), so all prompt variants for a part stay in one split.")
    print("- If you only generated 1600 parts per family, val/test may be empty unless you generate more parts (e.g., +400).")
    print("- If you see rows with note=NO_PROMPT_MATCH, we’ll adjust ID/index matching.")


if __name__ == "__main__":
    main()
