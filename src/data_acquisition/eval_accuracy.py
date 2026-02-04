import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


REQUIRED_KEYS = {
    "shaft": ["SHAFT_OD", "SHAFT_L", "BORE_D", "CHAMFER_L", "CHAMFER_ANG"],
    "bearing_sleeve": ["SLEEVE_OD", "SLEEVE_L", "BORE_D", "CHAMFER_L", "CHAMFER_ANG"],
    "bearing_flanged": ["FLANGE_OD", "FLANGE_THK", "BORE_D", "BOLT_D", "BOLT_PCD", "BOLT_COUNT", "BOLT_DEPTH"],
}


# ---------------------------
# JSON extraction helpers
# ---------------------------
def extract_all_json_dicts(text: str):
    """Extract any JSON objects (dicts) from a raw string."""
    dec = json.JSONDecoder()
    out = []
    i = 0
    while True:
        i = text.find("{", i)
        if i == -1:
            break
        try:
            obj, n = dec.raw_decode(text[i:])
            if isinstance(obj, dict):
                out.append(obj)
            i += max(1, n)
        except Exception:
            i += 1
    return out


def build_pred_from_text(out_text: str):
    """
    Build a prediction dict: {"family": <str>, "template_inputs": <dict>}
    from raw generation text that may contain extra JSON objects/noise.
    """
    objs = extract_all_json_dicts(out_text)
    if not objs:
        return None

    family = None
    for o in objs:
        if isinstance(o.get("family"), str):
            family = o["family"]
            break

    template_inputs = {}
    for o in objs:
        ti = o.get("template_inputs")
        if isinstance(ti, dict):
            template_inputs.update(ti)

    # fallback: if no wrapper exists, merge raw dicts
    if not template_inputs:
        for o in objs:
            if isinstance(o, dict):
                template_inputs.update(o)

    if not isinstance(family, str) or not template_inputs:
        return None

    return {"family": family, "template_inputs": template_inputs}


# ---------------------------
# Numeric handling helpers
# ---------------------------
def to_number(x):
    """Convert ints/floats or numeric strings to float. Return None if not numeric."""
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return float(x)
    if isinstance(x, str):
        try:
            # handle comma decimals too
            return float(x.strip().replace(",", "."))
        except Exception:
            return None
    return None


def within_tol(pred, gold, abs_tol=0.0, rel_tol=0.0):
    diff = abs(float(pred) - float(gold))
    return diff <= abs_tol + rel_tol * abs(float(gold))


def tol_for_key(key: str, mm_tol: float, angle_tol_deg: float):
    if "ANG" in key or key.endswith("_ANG"):
        return angle_tol_deg
    return mm_tol


def _safe_div(a: float, b: float) -> float:
    return a / b if b != 0 else 0.0

def compute_prf1_per_class(golds, preds, labels):
    """
    golds/preds: lists of label strings
    labels: list of unique labels to report
    Returns: dict[label] = (precision, recall, f1), plus macro averages
    """
    # Confusion components per class
    tp = {l: 0 for l in labels}
    fp = {l: 0 for l in labels}
    fn = {l: 0 for l in labels}

    for g, p in zip(golds, preds):
        for l in labels:
            if p == l and g == l:
                tp[l] += 1
            elif p == l and g != l:
                fp[l] += 1
            elif p != l and g == l:
                fn[l] += 1

    per_class = {}
    p_list, r_list, f1_list = [], [], []
    for l in labels:
        precision = _safe_div(tp[l], tp[l] + fp[l])
        recall = _safe_div(tp[l], tp[l] + fn[l])
        f1 = _safe_div(2 * precision * recall, precision + recall) if (precision + recall) else 0.0
        per_class[l] = (precision, recall, f1)
        p_list.append(precision)
        r_list.append(recall)
        f1_list.append(f1)

    macro_p = sum(p_list) / max(1, len(p_list))
    macro_r = sum(r_list) / max(1, len(r_list))
    macro_f1 = sum(f1_list) / max(1, len(f1_list))
    return per_class, (macro_p, macro_r, macro_f1)


# ---------------------------
# Core evaluation
# ---------------------------
def evaluate(
    model_dir: str,
    data_jsonl: str,
    max_new_tokens: int,
    temperature: float,
    rel_tol: float,
    angle_tol_deg: float,
    tolerances_mm: list[float],
    chosen_abs_tol: float,
    max_samples: int | None,
    print_every: int,
):
    print(f"[eval] model_dir: {model_dir}")
    print(f"[eval] data_jsonl: {data_jsonl}")
    print(f"[eval] max_new_tokens={max_new_tokens}, temperature={temperature}, rel_tol={rel_tol}, angle_tol_deg={angle_tol_deg}")
    print(f"[eval] tolerances_mm={tolerances_mm}")
    print(f"[eval] chosen_abs_tol={chosen_abs_tol}")
    if max_samples is not None:
        print(f"[eval] max_samples={max_samples}")
    print()

    path = Path(data_jsonl)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if max_samples is not None:
        rows = rows[:max_samples]

    total = len(rows)

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0),
        num_beams=1,
        pad_token_id=tok.eos_token_id,
        repetition_penalty=1.05,
    )
    if temperature > 0:
        gen_kwargs["temperature"] = temperature

    # Basic metrics
    parse_ok = 0
    family_ok = 0
    keys_ok = 0

    # For family classification metrics (precision/recall/F1)
    gold_families = []
    pred_families = []

    # Chosen tolerance metrics
    sample_correct = 0
    field_correct = 0
    field_total = 0

    # Multi-tolerance metrics
    field_correct_at = {t: 0 for t in tolerances_mm}
    field_total_at = {t: 0 for t in tolerances_mm}
    sample_correct_at = {t: 0 for t in tolerances_mm}

    t_start = time.perf_counter()

    for idx, r in enumerate(rows, start=1):
        prompt = r["prompt"]
        gold_family = r["family"]
        gold_inputs = r.get("template_inputs", {})

        prefix = "### Instruction:\n" + prompt.strip() + "\n\n### Response:\n"
        enc = tok(prefix, return_tensors="pt").to(device)

        with torch.no_grad():
            if device == "cuda":
                torch.cuda.synchronize()
            gen = model.generate(**enc, **gen_kwargs)
            if device == "cuda":
                torch.cuda.synchronize()

        out_text = tok.decode(gen[0], skip_special_tokens=True)
        pred = build_pred_from_text(out_text)

        # record gold always
        gold_families.append(gold_family)

        if pred is None:
            # treat unparsed output as "invalid" prediction for family metrics
            pred_families.append("invalid")
        else:
            parse_ok += 1
            pred_family = pred["family"]
            pred_families.append(pred_family)

            if pred_family == gold_family:
                family_ok += 1

            if pred["family"] == gold_family:
                family_ok += 1

                req = REQUIRED_KEYS.get(gold_family, list(gold_inputs.keys()))
                pred_inputs = pred["template_inputs"]

                if all(k in pred_inputs for k in req):
                    keys_ok += 1

                    # ---- multi-tolerance: track if the whole sample passes for each t
                    sample_ok_at = {t: True for t in tolerances_mm}

                    # ---- chosen tolerance: all fields must pass
                    sample_ok_chosen = True

                    for k in req:
                        gv = gold_inputs.get(k)
                        pv = pred_inputs.get(k)

                        gv_num = to_number(gv)
                        pv_num = to_number(pv)

                        # multi-tolerance field accounting
                        for t_mm in tolerances_mm:
                            field_total_at[t_mm] += 1

                        # chosen field accounting
                        field_total += 1

                        # numeric compare if possible
                        if gv_num is not None and pv_num is not None:
                            # COUNT should be integer-like
                            if k.endswith("COUNT"):
                                gv_num_cmp = int(round(gv_num))
                                pv_num_cmp = int(round(pv_num))
                            else:
                                gv_num_cmp = gv_num
                                pv_num_cmp = pv_num

                            # chosen tolerance per key
                            k_abs_tol_chosen = tol_for_key(k, chosen_abs_tol, angle_tol_deg)
                            if within_tol(pv_num_cmp, gv_num_cmp, abs_tol=k_abs_tol_chosen, rel_tol=rel_tol):
                                field_correct += 1
                            else:
                                sample_ok_chosen = False

                            # multi-tolerance
                            for t_mm in tolerances_mm:
                                k_abs_tol = tol_for_key(k, t_mm, angle_tol_deg)
                                if within_tol(pv_num_cmp, gv_num_cmp, abs_tol=k_abs_tol, rel_tol=rel_tol):
                                    field_correct_at[t_mm] += 1
                                else:
                                    sample_ok_at[t_mm] = False

                        else:
                            # non-numeric fallback: exact match
                            if pv == gv:
                                field_correct += 1
                                for t_mm in tolerances_mm:
                                    field_correct_at[t_mm] += 1
                            else:
                                sample_ok_chosen = False
                                for t_mm in tolerances_mm:
                                    sample_ok_at[t_mm] = False

                    if sample_ok_chosen:
                        sample_correct += 1

                    for t_mm in tolerances_mm:
                        if sample_ok_at[t_mm]:
                            sample_correct_at[t_mm] += 1

        # periodic progress
        if print_every > 0 and (idx % print_every == 0 or idx == total):
            elapsed = time.perf_counter() - t_start
            per_sample = elapsed / max(1, idx)
            eta = per_sample * (total - idx)

            parse_rate = 100.0 * parse_ok / max(1, idx)
            fam_rate = 100.0 * family_ok / max(1, idx)
            keys_rate = 100.0 * keys_ok / max(1, idx)
            sample_rate = 100.0 * sample_correct / max(1, idx)

            fam_given_parse = 100.0 * family_ok / max(1, parse_ok)
            keys_given_family = 100.0 * keys_ok / max(1, family_ok)
            sample_given_keys = 100.0 * sample_correct / max(1, keys_ok)

            print(
                f"[{idx}/{total}] "
                f"parse={parse_rate:.1f}% family={fam_rate:.1f}% keys={keys_rate:.1f}% sample={sample_rate:.2f}% | "
                f"(family|parse={fam_given_parse:.1f}%, keys|family={keys_given_family:.1f}%, sample|keys={sample_given_keys:.2f}%) | "
                f"{per_sample:.2f}s/sample ETA {eta/60:.1f}m",
                flush=True,
            )

    # Final summary
    print("\n=== Final Summary ===")
    print(f"total_samples: {total}")
    print(f"json_parse_rate: {100*parse_ok/max(1,total):.2f}%")
    print(f"family_accuracy: {100*family_ok/max(1,total):.2f}%")
        # ---- F1 / precision / recall for family classification
    # Labels to report: known families + optional "invalid" if it appeared
    base_labels = sorted(set([x for x in gold_families if isinstance(x, str)]))
    if "invalid" in pred_families and "invalid" not in base_labels:
        labels = base_labels + ["invalid"]
    else:
        labels = base_labels

    per_class, (macro_p, macro_r, macro_f1) = compute_prf1_per_class(gold_families, pred_families, labels)

    print("\nFamily classification metrics:")
    print(f"precision_macro: {100*macro_p:.2f}%")
    print(f"recall_macro:    {100*macro_r:.2f}%")
    print(f"f1_macro:        {100*macro_f1:.2f}%")
    print("per_class (precision / recall / f1):")
    for l in labels:
        p, r, f1 = per_class[l]
        print(f"  {l:16s} {100*p:6.2f}%  {100*r:6.2f}%  {100*f1:6.2f}%")

    print(f"required_keys_present: {100*keys_ok/max(1,total):.2f}%")

    # chosen tolerance accuracy
    print(f"chosen_abs_tol_mm: {chosen_abs_tol}")
    print(f"angle_abs_tol_deg: {angle_tol_deg}")
    print(f"field_accuracy(@chosen_tol): {100*field_correct/max(1,field_total):.2f}%")
    print(f"sample_accuracy(@chosen_tol): {100*sample_correct/max(1,total):.2f}%")

    # accuracy vs tolerance table (scored on all total samples, but only meaningful once keys are present)
    print("\nAccuracy vs tolerance (mm):")
    for t_mm in tolerances_mm:
        fa = 100 * field_correct_at[t_mm] / max(1, field_total_at[t_mm])
        sa = 100 * sample_correct_at[t_mm] / max(1, total)
        print(f"  ±{t_mm:>5} mm | field_acc={fa:6.2f}% | sample_acc={sa:6.2f}% (angle_tol=±{angle_tol_deg}°)")

    # practical note
    if keys_ok == total and sample_correct == 0:
        print("\n[note] Keys are always present, but strict sample accuracy is still low.")
        print("       This typically means your model predictions are close but not within the chosen tolerance on at least one field.")
        print("       Check the tolerance table above to pick a defendable margin for your report.")


def parse_tol_list(s: str) -> list[float]:
    # e.g. "0.1,0.5,1,2,5"
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evaluate prompt->JSON CAD parameter model accuracy on a JSONL dataset.")
    ap.add_argument("--model_dir", required=True, help="HF model name (e.g. gpt2) or path to trained model folder.")
    ap.add_argument("--data_jsonl", required=True, help="Path to processed test.jsonl/val.jsonl.")
    ap.add_argument("--max_new_tokens", type=int, default=180)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--rel_tol", type=float, default=0.0)
    ap.add_argument("--angle_tol_deg", type=float, default=1.0)
    ap.add_argument("--tolerances_mm", type=str, default="0.1,0.25,0.5,1,2,5", help="Comma-separated tolerances in mm.")
    ap.add_argument("--chosen_abs_tol", type=float, default=0.5, help="The tolerance used for the main sample/field accuracy print.")
    ap.add_argument("--max_samples", type=int, default=None, help="Evaluate only first N samples for faster iteration.")
    ap.add_argument("--print_every", type=int, default=25, help="Print running stats every N samples. Set 0 to disable.")
    args = ap.parse_args()

    tolerances = parse_tol_list(args.tolerances_mm)

    evaluate(
        model_dir=args.model_dir,
        data_jsonl=args.data_jsonl,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        rel_tol=args.rel_tol,
        angle_tol_deg=args.angle_tol_deg,
        tolerances_mm=tolerances,
        chosen_abs_tol=args.chosen_abs_tol,
        max_samples=args.max_samples,
        print_every=args.print_every,
    )
