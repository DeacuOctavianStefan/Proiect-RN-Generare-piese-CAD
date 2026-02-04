import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional


def norm_family_folder(family: str) -> str:
    f = family.strip().lower()
    if f in {"shaft"}:
        return "shaft"
    if f in {"sleeve", "bearing_sleeve"}:
        return "sleeve"
    if f in {"flange", "bearing_flanged", "bearing_flanged_simple", "bearing_flanged_simple_flange"}:
        return "flange"
    # fallback: use raw family
    return f.replace(" ", "_")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def run_one(
    python_exe: str,
    generator: Path,
    repo_root: Path,
    jsonl: Path,
    index: int,
    out_dir: Path,
    out_name: str,
    visible: int,
    verbose: int,
    strict: int,
    extra_args: Optional[list[str]] = None,
) -> tuple[int, str]:
    cmd = [
        python_exe,
        str(generator),
        "--repo_root", str(repo_root),
        "--jsonl", str(jsonl),
        "--index", str(index),
        "--out_dir", str(out_dir),
        "--out_name", out_name,
        "--visible", str(visible),
        "--verbose", str(verbose),
        "--strict", str(strict),
    ]
    if extra_args:
        cmd.extend(extra_args)

    # Run and capture output for logging
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return proc.returncode, proc.stdout


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch-generate SolidWorks parts from JSONL sweeps.")
    ap.add_argument("--repo_root", required=True, help="Path to repo root (e.g., D:\\Proiect-RN-Generare-piese-CAD-main)")
    ap.add_argument("--generator", required=True, help="Path to generator script (e.g., src\\data_acquisition\\generate_sldprt_from_json_dim_v3_6_1.py)")
    ap.add_argument("--jsonl", required=True, help="Path to input JSONL (e.g., data\\raw\\generated_sweep_valid\\shaft.jsonl)")
    ap.add_argument("--family", required=True, help="Family label used for folder naming (shaft | sleeve | flange | bearing_sleeve | bearing_flanged)")
    ap.add_argument("--split", required=True, choices=["train", "validation", "test"], help="Dataset split folder under data/")
    ap.add_argument("--count", type=int, required=True, help="How many parts to generate in this run")
    ap.add_argument("--start_index", type=int, default=0, help="Start index in the JSONL")
    ap.add_argument("--visible", type=int, default=0, help="1 to show SolidWorks, 0 to hide")
    ap.add_argument("--verbose", type=int, default=0, help="Generator verbosity")
    ap.add_argument("--strict", type=int, default=1, help="Generator strict mode (1 recommended)")
    ap.add_argument("--python", default=sys.executable, help="Python executable to use")
    ap.add_argument("--skip_existing", type=int, default=1, help="Skip if output file already exists (1=yes, 0=no)")
    ap.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between runs (sometimes helps SolidWorks stability)")
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    generator = (repo_root / args.generator).resolve() if not Path(args.generator).is_absolute() else Path(args.generator).resolve()
    jsonl = (repo_root / args.jsonl).resolve() if not Path(args.jsonl).is_absolute() else Path(args.jsonl).resolve()

    if not repo_root.exists():
        raise FileNotFoundError(f"repo_root not found: {repo_root}")
    if not generator.exists():
        raise FileNotFoundError(f"generator script not found: {generator}")
    if not jsonl.exists():
        raise FileNotFoundError(f"jsonl not found: {jsonl}")

    folder_family = norm_family_folder(args.family)
    out_dir = repo_root / "data" / args.split / "parts" / folder_family
    ensure_dir(out_dir)

    manifest = out_dir / "manifest.csv"
    new_file = not manifest.exists()

    with manifest.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["timestamp", "index", "out_name", "out_path", "status", "returncode", "note"])

        start = args.start_index
        end = start + args.count

        print(f"[batch] repo_root={repo_root}")
        print(f"[batch] jsonl={jsonl}")
        print(f"[batch] generator={generator}")
        print(f"[batch] split={args.split} family_folder={folder_family}")
        print(f"[batch] out_dir={out_dir}")
        print(f"[batch] indices: {start}..{end-1}")

        for idx in range(start, end):
            out_name = f"{folder_family}_{idx:06d}.SLDPRT"
            out_path = out_dir / out_name

            if args.skip_existing and out_path.exists():
                w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), idx, out_name, str(out_path), "SKIP_EXISTS", 0, ""])
                print(f"[skip] idx={idx} exists: {out_name}")
                continue

            rc, log = run_one(
                python_exe=args.python,
                generator=generator,
                repo_root=repo_root,
                jsonl=jsonl,
                index=idx,
                out_dir=out_dir,
                out_name=out_name,
                visible=args.visible,
                verbose=args.verbose,
                strict=args.strict,
                extra_args=None,
            )

            status = "OK" if rc == 0 and out_path.exists() else "FAIL"
            note = ""
            if status == "FAIL":
                # Keep logs smaller in CSV; store full log to a .log file
                log_path = out_dir / f"{folder_family}_{idx:06d}.log"
                log_path.write_text(log, encoding="utf-8", errors="replace")
                note = f"see {log_path.name}"

            w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), idx, out_name, str(out_path), status, rc, note])
            f.flush()

            print(f"[{status.lower()}] idx={idx} rc={rc} out={out_name}")
            if status == "FAIL":
                print("  -> wrote log:", note)

            if args.sleep > 0:
                time.sleep(args.sleep)


if __name__ == "__main__":
    main()
