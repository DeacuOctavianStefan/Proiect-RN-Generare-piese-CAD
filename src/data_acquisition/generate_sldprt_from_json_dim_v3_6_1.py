"""
generate_sldprt_from_json_dim_v3_3.py

Open a SolidWorks part template (.SLDPRT), set parameters from a JSONL record,
rebuild, and save a new .SLDPRT.

Supports:
1) Equation-driven (recommended): sets SolidWorks Global Variables via EquationMgr.
2) Dimension-driven fallback: sets named dimensions directly (DIM_MAP) if EquationMgr
   is not available.

Record formats supported (first match wins):
- record["template_inputs"] (used by data/raw/generated_sweep/*.jsonl)
- record["target"]["variables"]
- record["variables"]
- record["params"]
- record["inputs"]

CLI is kept compatible with your batch scripts.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

import pywintypes
import pythoncom
import win32com.client


# ----------------------------
# SolidWorks constants (minimal)
# ----------------------------
# Document types (swDocumentTypes_e)
swDocPART = 1

# Open doc options (swOpenDocOptions_e)
swOpenDocOptions_Silent = 1

# Save options (swSaveAsOptions_e)
swSaveAsOptions_Silent = 1
swSaveAsOptions_Copy = 2


# ----------------------------
# Family -> template path map (relative to repo root)
# ----------------------------
DEFAULT_TEMPLATES = {
    "bearing_flanged": "docs/templates/bearing_flanged_simple_flange_template.SLDPRT",
    "bearing_sleeve": "docs/templates/bearing_sleeve_bushing_template.SLDPRT",
    "shaft": "docs/templates/shaft_simple_bored_template.SLDPRT",
}

# Optional variable aliases (json_key -> template_global_var_name)
ALIASES_BY_FAMILY = {
    # You renamed the template GV from BOLT_PCD to BOLT_R
    "bearing_flanged": {"BOLT_PCD": "BOLT_R"},
}

# Fallback dimension mapping (only used if EquationMgr can't be accessed)
DIM_MAP = {
    "bearing_flanged": {
        "FLANGE_OD": "D1@Sketch1",
        "FLANGE_THK": "D1@Boss-Extrude1",
        "BORE_D": "D1@Sketch2",
        "BOLT_D": "Thru Hole Dia.@Sketch5",
        "BOLT_R": "D1@Sketch3",
        "BOLT_PCD": "D1@Sketch3",
        "BOLT_COUNT": "D1@CirPattern1",
        "BOLT_DEPTH": "Thru Hole Dep.@Sketch5",
    },
    "bearing_sleeve": {
        # NOTE: These dimension names are best-effort guesses.
        # If you ever hit the fallback path, run with --verbose 1 and adjust to match your template.
        "SLEEVE_OD": "D1@Sketch1",
        "SLEEVE_L": "D1@Boss-Extrude1",
        "BORE_D": "D1@Sketch2",
        "CHAMFER_D": "D1@Chamfer1",
        "CHAMFER_ANG": "D2@Chamfer1",
    },
    "shaft": {
        "SHAFT_OD": "D1@Sketch1",
        "SHAFT_L": "D1@Boss-Extrude1",
        "BORE_D": "D1@Sketch2",
        # Chamfer length (some users prefer CHAMFER_L). Keep CHAMFER_D for backward compatibility.
        "CHAMFER_D": "D1@Chamfer1",
        "CHAMFER_L": "D1@Chamfer1",
        "CHAMFER_ANG": "D2@Chamfer1",
    },
}


# ----------------------------
# JSON helpers
# ----------------------------

def get_param_system_value(param) -> float:
    """
    Safely read a SolidWorks parameter SystemValue (meters/radians).
    Some COM objects expose it slightly differently; this keeps it robust.
    """
    try:
        return float(param.SystemValue)
    except Exception:
        try:
            # Some SW versions expose GetSystemValue2; keep best-effort.
            return float(param.GetSystemValue2(""))
        except Exception:
            return float("nan")

def load_jsonl_record(path: Path, index: int | None = None, rec_id: str | None = None) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSONL not found: {path}")
    if index is None and rec_id is None:
        raise ValueError("Provide either --index or --id")

    with path.open("r", encoding="utf-8") as f:
        if rec_id is not None:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if str(r.get("id")) == rec_id:
                    return r
            raise KeyError(f"Record id not found: {rec_id}")

        for i, line in enumerate(f):
            if i == index:
                if not line.strip():
                    raise ValueError(f"Empty line at index {index}")
                return json.loads(line)

    raise IndexError(f"Index out of range: {index}")


def extract_vars(record: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Returns (vars_dict_or_None, where_string_for_debug)
    """
    if isinstance(record.get("template_inputs"), dict):
        return record["template_inputs"], "template_inputs"

    tgt = record.get("target")
    if isinstance(tgt, dict) and isinstance(tgt.get("variables"), dict):
        return tgt["variables"], "target.variables"

    for k in ("variables", "params", "inputs"):
        if isinstance(record.get(k), dict):
            return record[k], k

    return None, ""


# ----------------------------
# Unit / RHS formatting
# ----------------------------
def rhs_for_var(name: str, value: Any, units_hint: str | None = None) -> str:
    """
    Convert a python value into a SolidWorks equation RHS string.
    - If value is already a string, it is returned unchanged (e.g., "25mm", "45deg", "2*pi").
    - Heuristics:
        * *_COUNT -> unitless integer
        * contains "ANG" -> degrees
        * else -> mm
    """
    if isinstance(value, str):
        return value.strip()

    if value is None:
        raise ValueError(f"{name}: value is None")

    if name.upper().endswith("_COUNT"):
        return str(int(round(float(value))))

    if "ANG" in name.upper():
        return f"{float(value)}deg"

    return f"{float(value)}mm"


# ----------------------------
# SolidWorks COM helpers
# ----------------------------
def _safe_get_prop(obj: Any, prop: str, default=None):
    """Get COM property that may appear as prop or prop() depending on binding."""
    if not hasattr(obj, prop):
        return default
    v = getattr(obj, prop)
    try:
        return v() if callable(v) else v
    except Exception:
        return default


def ensure_sw_app(visible: bool = False):
    # SolidWorks expects STA COM apartment
    try:
        pythoncom.CoInitializeEx(pythoncom.COINIT_APARTMENTTHREADED)
    except Exception:
        # already initialized in this thread
        pass

    # Avoid EnsureDispatch (triggers makepy issue on some installs)
    try:
        sw = win32com.client.DispatchEx("SldWorks.Application")
    except Exception:
        sw = win32com.client.Dispatch("SldWorks.Application")

    # Set visibility early
    try:
        sw.Visible = bool(visible)
    except Exception:
        pass

    # Reduce UI prompts blocking automation
    for attr, val in (("UserControl", True), ("CommandInProgress", True)):
        try:
            setattr(sw, attr, val)
        except Exception:
            pass

    # Warm up: RevisionNumber is commonly a PROPERTY (string), not a callable.
    # So we must NOT do sw.RevisionNumber()
    last_exc = None
    for _ in range(60):
        try:
            _ = _safe_get_prop(sw, "RevisionNumber", default="")
            return sw
        except pywintypes.com_error as e:
            last_exc = e
            time.sleep(0.25)

    # If still not stable, return it anyway, but caller may hit retries.
    return sw


def open_part(sw_app: Any, template_path: Path) -> Tuple[Any, int, int]:
    template_path = Path(template_path).resolve()
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")

    errors = win32com.client.VARIANT(pythoncom.VT_BYREF | pythoncom.VT_I4, 0)
    warns = win32com.client.VARIANT(pythoncom.VT_BYREF | pythoncom.VT_I4, 0)

    last_exc = None
    for attempt in range(1, 11):
        try:
            model = sw_app.OpenDoc6(
                str(template_path),
                swDocPART,
                swOpenDocOptions_Silent,
                "",
                errors,
                warns,
            )
            if model is None:
                model = sw_app.ActiveDoc
            if model is None:
                raise RuntimeError("OpenDoc6 returned None and ActiveDoc is None")

            # Cast if possible (ok if it fails)
            try:
                model = win32com.client.CastTo(model, "ModelDoc2")
            except Exception:
                pass

            return model, int(errors.value), int(warns.value)

        except pywintypes.com_error as e:
            last_exc = e
            # RPC failures / call rejected / server busy often resolve with short backoff
            time.sleep(0.5 * attempt)

    raise RuntimeError(f"OpenDoc6 failed after retries. Last error: {last_exc}")


def get_equation_mgr(model: Any) -> Any | None:
    """
    Try several access patterns; SolidWorks exposure can differ by binding/wrappers.
    """
    candidates = [
        ("GetEquationMgr", True),
        ("EquationMgr", False),
        ("IGetEquationMgr", True),
        ("GetEquationMgr2", True),
    ]

    for attr, call in candidates:
        try:
            if not hasattr(model, attr):
                continue
            obj = getattr(model, attr)
            eq = obj() if (call and callable(obj)) else obj
            if eq is not None:
                return eq
        except pywintypes.com_error:
            continue
        except Exception:
            continue

    try:
        model2 = win32com.client.CastTo(model, "ModelDoc2")
        for attr, call in candidates:
            try:
                if not hasattr(model2, attr):
                    continue
                obj = getattr(model2, attr)
                eq = obj() if (call and callable(obj)) else obj
                if eq is not None:
                    return eq
            except pywintypes.com_error:
                continue
            except Exception:
                continue
    except Exception:
        pass

    return None


def eq_count(eq_mgr: Any) -> int:
    for attr in ("GetCount", "Count"):
        if hasattr(eq_mgr, attr):
            v = getattr(eq_mgr, attr)
            try:
                return int(v() if callable(v) else v)
            except Exception:
                continue
    raise RuntimeError("Cannot determine EquationMgr count.")


def eq_get(eq_mgr: Any, i: int) -> str:
    # Some bindings expose Equation as indexable property
    try:
        return str(eq_mgr.Equation[i])
    except Exception:
        pass
    # Some expose as a callable
    try:
        return str(eq_mgr.Equation(i))
    except Exception:
        pass
    # Fallback: raw COM invoke
    dispid = eq_mgr._oleobj_.GetIDsOfNames("Equation")
    return str(eq_mgr._oleobj_.Invoke(dispid, 0, pythoncom.DISPATCH_PROPERTYGET, True, (i,)))


def eq_set(eq_mgr: Any, i: int, equation_str: str) -> None:
    try:
        eq_mgr.Equation[i] = equation_str
        return
    except Exception:
        pass
    try:
        eq_mgr.Equation(i, equation_str)
        return
    except Exception:
        pass
    dispid = eq_mgr._oleobj_.GetIDsOfNames("Equation")
    eq_mgr._oleobj_.Invoke(dispid, 0, pythoncom.DISPATCH_PROPERTYPUT, False, (i, equation_str))


def rebuild(model: Any) -> None:
    # Pick a safe rebuild method depending on what binding exposes
    for name, args in (("ForceRebuild3", (False,)), ("EditRebuild3", ()), ("Rebuild", (0,))):
        if not hasattr(model, name):
            continue
        fn = getattr(model, name)
        try:
            if callable(fn):
                fn(*args)
            return
        except Exception:
            continue



def set_param_system_value(param: Any, new_value: float) -> None:
    """Robustly set a SOLIDWORKS dimension/parameter value.

    Some SW COM bindings do not persist assignments to .SystemValue reliably.
    Prefer SetSystemValue3 when available (sets the value in the current configuration).
    """
    # swSetValueInConfiguration_e: 2 = This configuration (commonly)
    THIS_CONFIG = 2
    if hasattr(param, "SetSystemValue3"):
        try:
            param.SetSystemValue3(float(new_value), THIS_CONFIG, "")
            return
        except Exception:
            pass
    if hasattr(param, "SetSystemValue2"):
        try:
            param.SetSystemValue2(float(new_value), THIS_CONFIG, "")
            return
        except Exception:
            pass
    # Fallback
    param.SystemValue = float(new_value)


def save_as(model: Any, out_path: Path) -> Tuple[bool, int, int]:
    """Save the active document to out_path.

    SolidWorks exposes different SaveAs* signatures depending on version / type library.
    With late-bound COM (pywin32), calling the wrong arity/order often results in:
      pywintypes.com_error: Type mismatch.

    We therefore:
      1) Prefer ModelDocExtension.SaveAs3 with the 6-arg signature:
         SaveAs3(FileName, Version, Options, ExportData, Errors, Warnings)
      2) Fall back to SaveAs4 if present (similar signature + more options)
      3) Fall back to ModelDoc2.SaveAs as a last resort.
    """
    ext = model.Extension
    errs = win32com.client.VARIANT(pythoncom.VT_BYREF | pythoncom.VT_I4, 0)
    warns = win32com.client.VARIANT(pythoncom.VT_BYREF | pythoncom.VT_I4, 0)

    version = 0  # current
    options = int(swSaveAsOptions_Silent) | int(swSaveAsOptions_Copy)
    export_data = None

    # Try SaveAs3 (6-arg)
    try:
        ok = ext.SaveAs3(str(out_path), version, options, export_data, errs, warns)
        return bool(ok), int(errs.value), int(warns.value)
    except Exception:
        pass

    # Try SaveAs4 if available (some SW versions expose SaveAs4)
    try:
        # SaveAs4(FileName, Version, Options, ExportData, Errors, Warnings)
        ok = ext.SaveAs4(str(out_path), version, options, export_data, errs, warns)
        return bool(ok), int(errs.value), int(warns.value)
    except Exception:
        pass

    # Last resort: ModelDoc2.SaveAs (older signature; may ignore errors/warnings)
    try:
        ok = model.SaveAs(str(out_path))
        return bool(ok), int(errs.value), int(warns.value)
    except Exception as e:
        raise e


# ----------------------------
# Parameter setting
# ----------------------------
def set_globals_via_equations(
    model: Any,
    family: str,
    vars_in: Dict[str, Any],
    units_hint: str | None,
    strict: bool,
    verbose: int,
) -> Tuple[int, Dict[str, Any]]:
    eq_mgr = get_equation_mgr(model)
    if eq_mgr is None:
        raise RuntimeError("Could not access EquationMgr from this document.")

    n = eq_count(eq_mgr)

    # Build map of "GLOBAL_VAR_NAME" -> equation index
    name_to_idx: Dict[str, int] = {}
    for i in range(n):
        s = eq_get(eq_mgr, i).strip()
        if s.startswith('"'):
            endq = s.find('"', 1)
            if endq > 1 and "=" in s[endq:]:
                name_to_idx[s[1:endq]] = i

    aliases = ALIASES_BY_FAMILY.get(family, {})
    unmapped: Dict[str, Any] = {}
    params_set = 0

    # Allow both "input key -> template GV" and the reverse, so either name works.
    rev_aliases = {v: k for k, v in aliases.items()}

    def resolve_name(input_key: str) -> str | None:
        candidates = []
        if input_key in aliases:
            candidates.append(aliases[input_key])
        candidates.append(input_key)
        if input_key in rev_aliases:
            candidates.append(rev_aliases[input_key])
        for cand in candidates:
            if cand in name_to_idx:
                return cand
        return None

    for k, v in vars_in.items():

        # First, try setting by linked-name parameter (if template uses Link Values and exposes it).
        link_param = try_get_param_by_key(model, k)
        if link_param is not None:
            if isinstance(v, str):
                if strict:
                    raise RuntimeError(f"Linked-parameter set requires numeric values. Got string for {k}: {v}")
                unmapped[k] = v
                continue

            k_up = k.upper()
            is_angle = k_up.endswith('_ANG') or k_up.endswith('_ANGLE') or k_up in {'CHAMFER_ANG'}
            is_count = k_up.endswith('_COUNT') or k_up.endswith('_QTY') or k_up in {'BOLT_COUNT'}

            if is_count:
                target_val = float(v)
            elif is_angle:
                target_val = float(v) * 3.141592653589793 / 180.0
            else:
                target_val = float(v) / 1000.0

            set_param_system_value(link_param, target_val)
            if verbose:
                try:
                    rb = float(get_param_system_value(link_param))
                    print(f"SET LINK {k} -> {link_param.GetNameForSelection()}  target={target_val}  readback={rb}")
                except Exception:
                    print(f"SET LINK {k} -> (linked param)  target={target_val}")
            params_set += 1
            continue

        gv_name = resolve_name(k)

        if gv_name is None:
            if strict:
                available = ", ".join(sorted(name_to_idx.keys())[:50])
                raise RuntimeError(
                    f"Global variable not found in template for input key: {k}\n"
                    f"Available (first 50): {available}"
                )
            unmapped[k] = v
            continue

        rhs = rhs_for_var(gv_name, v, units_hint)
        eq_str = f'"{gv_name}"={rhs}'
        idx = name_to_idx[gv_name]

        if verbose:
            before = eq_get(eq_mgr, idx)
            print(f"SET GV {k} @Eq[{idx}]  before: {before}  ->  {eq_str}")

        eq_set(eq_mgr, idx, eq_str)
        params_set += 1

    try:
        if hasattr(eq_mgr, "EvaluateAll") and callable(eq_mgr.EvaluateAll):
            eq_mgr.EvaluateAll()
    except Exception:
        pass

    rebuild(model)
    return params_set, unmapped



def try_get_param_by_key(model: Any, key: str) -> Any:
    """
    Try to access a SOLIDWORKS Parameter by a symbolic key name.

    This is useful when you used 'Link Values' in the template and the linked name
    is exposed as a parameter (often as KEY or "KEY").
    """
    if not key:
        return None
    candidates = [key, f'"{key}"']
    for cand in candidates:
        try:
            p = model.Parameter(cand)
            if p is not None:
                return p
        except Exception:
            continue
    return None

def find_linked_param(model, family: str, key: str):
    """
    Try to find a parameter created by 'Link Values' or similar, using common SolidWorks naming forms:
      KEY
      "KEY"
      KEY@Sketch1, "KEY"@Sketch1, etc.

    Returns COM parameter object or None.
    """
    # Common owners per family (adjust if you renamed features/sketches)
    owners_by_family = {
        "shaft": ["Sketch1", "Boss-Extrude1", "Sketch2", "Chamfer1"],
        "bearing_sleeve": ["Sketch1", "Boss-Extrude1", "Sketch2", "Chamfer1"],
        "bearing_flanged": ["Sketch1", "Sketch2", "Sketch3", "Boss-Extrude1", "Cut-Extrude1", "Cut-Extrude2", "CirPattern1"],
    }

    owners = owners_by_family.get(family, [])
    candidates = [key, f'"{key}"']

    # Try KEY@Owner variants (this is the big missing piece)
    for owner in owners:
        candidates.append(f"{key}@{owner}")
        candidates.append(f'"{key}"@{owner}')

    # Try lookup
    for name in candidates:
        try:
            p = model.Parameter(name)
            if p is not None:
                return p
        except Exception:
            pass

    return None


def set_dimensions_fallback(
    model: Any,
    family: str,
    vars_in: Dict[str, Any],
    strict: bool,
    verbose: int,
) -> Tuple[int, Dict[str, Any]]:
    dim_map = DIM_MAP.get(family, {})
    unmapped: Dict[str, Any] = {}
    params_set = 0

    for k, v in vars_in.items():
        # 1) Prefer linked parameters (Link Values) if they exist
        link_param = find_linked_param(model, family, k)

        # Support your rename: CHAMFER_D <-> CHAMFER_L
        if link_param is None and k == "CHAMFER_D":
            link_param = find_linked_param(model, family, "CHAMFER_L")
        if link_param is None and k == "CHAMFER_L":
            link_param = find_linked_param(model, family, "CHAMFER_D")

        # Common conversion logic (same as below)
        if isinstance(v, str):
            if strict:
                raise RuntimeError(f"Dimension fallback requires numeric values. Got string for {k}: {v}")
            unmapped[k] = v
            continue

        k_up = k.upper()
        is_angle = k_up.endswith("_ANG") or k_up.endswith("_ANGLE") or k_up in {"CHAMFER_ANG", "CHAMFER_ANGLE"}
        is_count = k_up.endswith("_COUNT") or k_up.endswith("_QTY") or k_up in {"BOLT_COUNT"}

        if is_count:
            target_val = float(v)  # unitless
        elif is_angle:
            target_val = float(v) * 3.141592653589793 / 180.0  # deg -> rad
        else:
            target_val = float(v) / 1000.0  # mm -> m

        if link_param is not None:
            set_param_system_value(link_param, target_val)
            params_set += 1
            if verbose:
                try:
                    nm = link_param.GetNameForSelection()
                except Exception:
                    nm = "(linked param)"
                rb = get_param_system_value(link_param)
                print(f"SET LINK {k} -> {nm}  target={target_val}  readback={rb}")
            continue

        # 2) Fall back to hard-coded dimension ids
        dim_name = dim_map.get(k)
        if dim_name is None:
            unmapped[k] = v
            continue

        # Hole Wizard naming variants (Dep./Depth, Dia./Diameter)
        candidates = [dim_name]
        if "Dep." in dim_name:
            candidates.append(dim_name.replace("Dep.", "Depth"))
        if "Depth" in dim_name:
            candidates.append(dim_name.replace("Depth", "Dep."))
        if "Dia." in dim_name:
            candidates.append(dim_name.replace("Dia.", "Diameter"))
        if "Diameter" in dim_name:
            candidates.append(dim_name.replace("Diameter", "Dia."))

        dim = None
        used_name = dim_name
        for cand in candidates:
            try:
                d = model.Parameter(cand)
            except Exception:
                d = None
            if d is not None:
                dim = d
                used_name = cand
                break

        if dim is None:
            if strict:
                raise RuntimeError(f"Dimension not found: {dim_name} (for key {k})")
            unmapped[k] = v
            continue

        set_param_system_value(dim, target_val)
        params_set += 1
        if verbose:
            rb = get_param_system_value(dim)
            print(f"SET DIM {k} -> {used_name}  target={target_val}  readback={rb}")

    rebuild(model)
    return params_set, unmapped


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, help="Path to jsonl file")
    ap.add_argument("--id", dest="rec_id", default=None, help="Record id (optional)")
    ap.add_argument("--index", type=int, default=None, help="Record index (0-based)")
    ap.add_argument("--repo_root", default=".", help="Repo root (default: current directory)")
    ap.add_argument("--out_dir", required=True, help="Output directory")
    ap.add_argument("--out_name", default=None, help="Override output filename (optional)")
    ap.add_argument("--visible", type=int, default=0, help="SolidWorks visible (0/1)")
    ap.add_argument("--strict", type=int, default=1, help="Strict mapping (0/1)")
    ap.add_argument("--verbose", type=int, default=0, help="Verbose logging (0/1)")
    args = ap.parse_args()

    jsonl_path = Path(args.repo_root) / args.jsonl if not Path(args.jsonl).is_absolute() else Path(args.jsonl)
    out_dir = Path(args.repo_root) / args.out_dir if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    record = load_jsonl_record(jsonl_path, index=args.index, rec_id=args.rec_id)

    family = record.get("family") or record.get("subtype") or record.get("type")
    if not family:
        print("Record missing 'family'. Keys:", list(record.keys()))
        sys.exit(2)

    vars_in, where = extract_vars(record)
    if vars_in is None:
        print("No variables found in record. Expected one of: target.variables / variables / params / inputs / template_inputs.")
        print("TOP LEVEL KEYS:", list(record.keys()))
        sys.exit(2)

    units_hint = record.get("units")
    template_rel = record.get("template") or DEFAULT_TEMPLATES.get(family)
    if not template_rel:
        print(f"No template path found for family '{family}'. Provide record['template'] or add to DEFAULT_TEMPLATES.")
        sys.exit(2)

    template_path = Path(args.repo_root) / template_rel if not Path(template_rel).is_absolute() else Path(template_rel)
    if not template_path.exists():
        print(f"Template not found: {template_path}")
        sys.exit(2)

    out_path = (out_dir / args.out_name) if args.out_name else (out_dir / f"{family}_{uuid.uuid4().hex[:8]}.SLDPRT")

    strict = bool(args.strict)
    verbose = int(args.verbose)

    sw_app = ensure_sw_app(visible=bool(args.visible))

    model, open_err, open_warn = open_part(sw_app, template_path)
    if model is None:
        print(f"Failed to open template. open_err={open_err} open_warn={open_warn}")
        sys.exit(1)

    # Dimension-driven mode (reliable via COM). We intentionally skip EquationMgr/global-variable setting
    # because SolidWorks 2026 + pywin32 may not expose EquationMgr consistently.
    params_set, unmapped = set_dimensions_fallback(model, family, vars_in, strict=strict, verbose=verbose)
    mode = "dimensions"

    # Force a rebuild so geometry updates before saving (critical when driving dimensions via COM)
    rebuild(model)
    try:
        if hasattr(model, "GraphicsRedraw2"):
            model.GraphicsRedraw2()
    except Exception:
        pass

    ok, save_err, save_warn = save_as(model, out_path)
    if not ok:
        print(f"SaveAs failed. errors={save_err} warnings={save_warn} path={out_path}")
        sys.exit(1)

    print("OK")
    print(f"mode: {mode}")
    print(f"family: {family}")
    print(f"template: {template_rel}")
    print(f"out: {out_path}")
    print(f"vars_from: {where}")
    print(f"params_set: {params_set}  unmapped_keys: {len(unmapped)}")
    print(f"open_err={open_err} open_warn={open_warn} save_err={save_err} save_warn={save_warn}")

    # Close doc when running headless
    try:
        if not bool(args.visible):
            title = _safe_get_prop(model, "GetTitle", default=None) or _safe_get_prop(model, "Title", default=None)
            if title:
                sw_app.CloseDoc(title)
    except Exception:
        pass


if __name__ == "__main__":
    main()