import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# SolidWorks is Windows-only COM automation
try:
    import pythoncom
    import win32com.client
except ImportError as e:
    raise SystemExit(
        "Missing pywin32. Install it in the SAME Python you run this script with:\n"
        "  pip install pywin32\n"
        "Also ensure Python bitness matches SolidWorks (usually 64-bit)."
    ) from e


# --- SolidWorks constants (avoid relying on generated constants) ---
SW_DOC_PART = 1
SW_OPEN_SILENT = 1
SW_SAVEAS_SILENT = 1

# Extension.SaveAs versions vary; this works for most installs:
# ext.SaveAs(path, SaveAsVersion, Options, ExportData, errors, warnings)
SAVEAS_VERSION_CURRENT = 0


def read_nth_jsonl(path: Path, index0: int) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            if i == index0:
                return json.loads(line)
    raise IndexError(f"Index {index0} out of range for {path}")


def read_by_id_jsonl(path: Path, wanted_id: str) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if str(rec.get("id")) == wanted_id:
                return rec
    raise KeyError(f"Record id not found: {wanted_id} in {path}")


def format_equation_value(var_name: str, value: Any) -> str:
    """
    Convert numeric value into a SolidWorks equation RHS string.
    - *_ANG -> degrees
    - *_COUNT -> unitless integer
    - otherwise -> mm
    """
    name = var_name.upper()
    if "COUNT" in name:
        return str(int(round(float(value))))
    if "ANG" in name:
        return f"{float(value)}deg"
    # default mm
    return f"{float(value)}mm"


def set_global_variables(model, inputs: Dict[str, Any]) -> Tuple[int, int]:
    """
    Update existing Global Variables in Tools->Equations.
    Returns (updated_count, missing_count).
    """
    eq_mgr = model.GetEquationMgr()
    n = eq_mgr.GetCount()

    # Map variable name -> equation index
    idx_map = {}
    for i in range(n):
        eq = eq_mgr.Equation(i)  # e.g. '"SHAFT_OD"=20mm'
        # Try to parse a leading quoted name
        if isinstance(eq, str) and eq.startswith('"'):
            end = eq.find('"', 1)
            if end > 1:
                name = eq[1:end]
                idx_map[name] = i

    updated = 0
    missing = 0

    for k, v in inputs.items():
        if k not in idx_map:
            missing += 1
            continue
    rhs = format_equation_value(k, v)
    eq_mgr.SetEquation(idx_map[k], f'"{k}"={rhs}')
    updated += 1


    # Force rebuild of equations
    # (EditRebuild3 afterwards will rebuild features too)
    try:
        eq_mgr.EvaluateAll()
    except Exception:
        pass

    return updated, missing


def ensure_abs_path(repo_root: Path, maybe_rel: str) -> Path:
    p = Path(maybe_rel)
    if p.is_absolute():
        return p
    return (repo_root / p).resolve()


def open_part(sw_app, path: Path):
    # Use byref ints for COM out params
    errs = pythoncom.VARIANT(pythoncom.VT_BYREF | pythoncom.VT_I4, 0)
    warns = pythoncom.VARIANT(pythoncom.VT_BYREF | pythoncom.VT_I4, 0)
    model = sw_app.OpenDoc6(str(path), SW_DOC_PART, SW_OPEN_SILENT, "", errs, warns)
    return model, int(errs.value), int(warns.value)


def save_as(model, out_path: Path) -> Tuple[bool, int, int]:
    ext = model.Extension
    errs = pythoncom.VARIANT(pythoncom.VT_BYREF | pythoncom.VT_I4, 0)
    warns = pythoncom.VARIANT(pythoncom.VT_BYREF | pythoncom.VT_I4, 0)

    ok = ext.SaveAs(
        str(out_path),
        SAVEAS_VERSION_CURRENT,
        SW_SAVEAS_SILENT,
        None,
        errs,
        warns,
    )
    return bool(ok), int(errs.value), int(warns.value)


def main():
    ap = argparse.ArgumentParser(description="Generate a .SLDPRT from a processed JSONL record using SolidWorks templates.")
    ap.add_argument("--jsonl", required=True, help="Input JSONL file (e.g., data/processed/test.jsonl).")
    ap.add_argument("--id", default="", help="Record id to generate (preferred).")
    ap.add_argument("--index", type=int, default=-1, help="0-based index in the JSONL if --id not provided.")
    ap.add_argument("--repo_root", default=".", help="Repo root (used to resolve relative template paths).")
    ap.add_argument("--out_dir", default="data/raw/generated_parts", help="Output folder for generated .SLDPRT.")
    ap.add_argument("--out_name", default="", help="Optional exact output filename (without path).")
    ap.add_argument("--visible", type=int, default=1, help="1 = show SolidWorks, 0 = run hidden.")
    ap.add_argument("--strict", type=int, default=1, help="1 = fail if any variable missing in template; 0 = allow missing.")
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    jsonl_path = ensure_abs_path(repo_root, args.jsonl)

    if args.id:
        rec = read_by_id_jsonl(jsonl_path, args.id)
    else:
        if args.index < 0:
            raise SystemExit("Provide either --id or --index >= 0")
        rec = read_nth_jsonl(jsonl_path, args.index)

    template_rel = rec.get("template")
    if not template_rel:
        raise SystemExit("Record missing 'template' field.")
    template_path = ensure_abs_path(repo_root, template_rel)

    # Processed dataset uses "target" as the template inputs
    inputs = rec.get("target") or rec.get("template_inputs")
    if not isinstance(inputs, dict) or not inputs:
        raise SystemExit("Record missing 'target' (template inputs).")

    family = rec.get("family", "part")
    subtype = rec.get("subtype", "base")

    out_dir = ensure_abs_path(repo_root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.out_name:
        out_name = args.out_name
        if not out_name.lower().endswith(".sldprt"):
            out_name += ".SLDPRT"
    else:
        # stable-ish filename
        rid = str(rec.get("id", "noid"))[:8]
        out_name = f"{family}_{subtype}_{rid}.SLDPRT"

    out_path = (out_dir / out_name).resolve()

    # --- COM automation ---
    pythoncom.CoInitialize()
    try:
        sw_app = win32com.client.DispatchEx("SldWorks.Application")
        sw_app.Visible = bool(args.visible)

        if not template_path.exists():
            raise SystemExit(f"Template not found: {template_path}")

        model, open_err, open_warn = open_part(sw_app, template_path)
        if model is None:
            raise SystemExit(f"Failed to open template. errors={open_err} warnings={open_warn}")

        updated, missing = set_global_variables(model, inputs)

        if args.strict and missing > 0:
            # Close doc and fail
            title = model.GetTitle()
            try:
                sw_app.CloseDoc(title)
            except Exception:
                pass
            raise SystemExit(
                f"Template is missing {missing} global variable(s). "
                f"Updated {updated}. Ensure Tools->Equations contains all keys in 'target'."
            )

        # Rebuild
        model.EditRebuild3()

        ok, save_err, save_warn = save_as(model, out_path)
        if not ok:
            raise SystemExit(f"SaveAs failed. errors={save_err} warnings={save_warn} path={out_path}")

        # Close the document to keep SolidWorks stable during batch runs
        title = model.GetTitle()
        try:
            sw_app.CloseDoc(title)
        except Exception:
            pass

        print("OK")
        print(f"template: {template_rel}")
        print(f"out: {out_path}")
        print(f"vars_updated: {updated}  vars_missing: {missing}")
        print(f"open_err={open_err} open_warn={open_warn} save_err={save_err} save_warn={save_warn}")

    finally:
        pythoncom.CoUninitialize()


if __name__ == "__main__":
    main()
