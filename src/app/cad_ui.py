import os
import subprocess
from pathlib import Path

import sys
import streamlit as st

# --- Defaults (adjust if you want) ---
DEFAULT_REPO_ROOT = r"D:\Proiect-RN-Generare-piese-CAD-main"
DEFAULT_MODEL_DIR = str(Path(DEFAULT_REPO_ROOT) / "runs" / "json_sft_gpt2_with_metrics_v2" / "final")
DEFAULT_OUT_DIR = "out/ui_parts"  # relative to repo_root

INFER_SCRIPT = "src/neural_network/infer_json.py"
SW_SCRIPT = "src/data_acquisition/generate_sldprt_from_json_dim_v3_6_1.py"


def run_cmd(cmd, cwd):
    """Run a command, stream output, and return (code, stdout, stderr)."""
    p = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        capture_output=True,
        shell=False,
    )
    return p.returncode, p.stdout, p.stderr


st.set_page_config(page_title="CAD Part Generator", layout="wide")
st.title("Prompt → JSON → SolidWorks Part (.SLDPRT)")

left, right = st.columns([1, 1])

with left:
    st.subheader("Inputs")

    repo_root = st.text_input("Repo root", value=DEFAULT_REPO_ROOT)
    repo_root_path = Path(repo_root).resolve()

    model_dir = st.text_input("Model dir (runs/<run>/final)", value=DEFAULT_MODEL_DIR)

    prompt = st.text_area("Prompt", height=180, placeholder="Describe the part you want...")

    max_new_tokens = st.number_input("max_new_tokens", min_value=32, max_value=512, value=180, step=8)
    temperature = st.number_input("temperature", min_value=0.0, max_value=2.0, value=0.0, step=0.1)

    out_jsonl = st.text_input("Output JSONL (relative to repo root)", value="out/predicted_one.jsonl")
    out_dir = st.text_input("Output parts directory (relative to repo root)", value=DEFAULT_OUT_DIR)

    visible = st.checkbox("Show SolidWorks UI (visible=1)", value=False)
    strict = st.checkbox("Strict mapping (strict=1)", value=True)
    verbose = st.checkbox("Verbose SolidWorks logging (verbose=1)", value=False)

    run_btn = st.button("Generate Part", type="primary", disabled=(not prompt.strip()))

with right:
    st.subheader("Output")
    status_box = st.empty()
    infer_out_box = st.empty()
    sw_out_box = st.empty()
    json_preview_box = st.empty()
    part_path_box = st.empty()

if run_btn:
    # Basic path checks
    if not repo_root_path.exists():
        st.error(f"Repo root does not exist: {repo_root_path}")
        st.stop()

    infer_path = repo_root_path / INFER_SCRIPT
    sw_path = repo_root_path / SW_SCRIPT
    if not infer_path.exists():
        st.error(f"Cannot find inference script: {infer_path}")
        st.stop()
    if not sw_path.exists():
        st.error(f"Cannot find SolidWorks generator script: {sw_path}")
        st.stop()

    out_jsonl_path = (repo_root_path / out_jsonl).resolve()
    out_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    out_parts_dir = (repo_root_path / out_dir).resolve()
    out_parts_dir.mkdir(parents=True, exist_ok=True)

    status_box.info("Running inference (prompt → JSONL)...")

    # 1) Run infer_json.py
    infer_cmd = [
        sys.executable,
        str(infer_path),
        "--repo_root", str(repo_root_path),
        "--model_dir", str(Path(model_dir).resolve() if Path(model_dir).exists() else model_dir),
        "--prompt", prompt,
        "--max_new_tokens", str(int(max_new_tokens)),
        "--temperature", str(float(temperature)),
        "--out_jsonl", str(out_jsonl_path.relative_to(repo_root_path)),
    ]

    code, out, err = run_cmd(infer_cmd, cwd=str(repo_root_path))
    infer_out_box.code(out + ("\n" + err if err else ""), language="text")

    if code != 0:
        status_box.error("Inference failed. See output above.")
        st.stop()

    if not out_jsonl_path.exists() or out_jsonl_path.stat().st_size == 0:
        status_box.error(f"Inference did not create JSONL: {out_jsonl_path}")
        st.stop()

    # Preview generated JSONL line
    line = out_jsonl_path.read_text(encoding="utf-8").splitlines()[0].strip()
    json_preview_box.code(line, language="json")

    status_box.info("Running SolidWorks generation (JSONL → .SLDPRT)...")

    # 2) Run SolidWorks generator on index 0
    sw_cmd = [
        sys.executable,
        str(sw_path),
        "--jsonl", str(out_jsonl_path),
        "--index", "0",
        "--repo_root", str(repo_root_path),
        "--out_dir", str(out_parts_dir),
        "--visible", "1" if visible else "0",
        "--strict", "1" if strict else "0",
        "--verbose", "1" if verbose else "0",
    ]

    code2, out2, err2 = run_cmd(sw_cmd, cwd=str(repo_root_path))
    sw_out_box.code(out2 + ("\n" + err2 if err2 else ""), language="text")

    if code2 != 0:
        status_box.error("SolidWorks generation failed. See output above.")
        st.stop()

    # Try to extract the saved part path from output line: "out: <path>"
    part_path = None
    for ln in (out2 + "\n" + err2).splitlines():
        if ln.strip().lower().startswith("out:"):
            part_path = ln.split(":", 1)[1].strip()
            break

    status_box.success("Done.")
    if part_path:
        part_path_box.success(f"Saved part: {part_path}")
    else:
        part_path_box.warning("Done, but could not detect output path. Check SolidWorks output above.")
