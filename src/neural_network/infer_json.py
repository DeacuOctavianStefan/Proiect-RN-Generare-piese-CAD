import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

REQUIRED_KEYS = {
    "shaft": ["SHAFT_OD", "SHAFT_L", "BORE_D", "CHAMFER_L", "CHAMFER_ANG"],
    "bearing_sleeve": ["SLEEVE_OD", "SLEEVE_L", "BORE_D", "CHAMFER_L", "CHAMFER_ANG"],
    "bearing_flanged": ["FLANGE_OD", "FLANGE_THK", "BORE_D", "BOLT_D", "BOLT_PCD", "BOLT_COUNT", "BOLT_DEPTH"],
}


def extract_first_json_object(text: str):
    decoder = json.JSONDecoder()
    i = 0
    while True:
        i = text.find("{", i)
        if i == -1:
            return None
        try:
            obj, end = decoder.raw_decode(text[i:])
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        i += 1

def extract_all_json_dicts(text: str):
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", required=True)
    ap.add_argument("--model_dir", required=True, help="Path to runs/<run_name>/final")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=180)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--out_jsonl", default="out/predicted_one.jsonl")
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    model_dir = Path(args.model_dir).resolve()
    out_path = (repo / args.out_jsonl).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    prefix = "### Instruction:\n" + args.prompt.strip() + "\n\n### Response:\n"
    enc = tok(prefix, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=(args.temperature > 0),
        num_beams=1,
        pad_token_id=tok.eos_token_id,
    )

    if args.temperature > 0:
        gen_kwargs["temperature"] = args.temperature
    gen_kwargs["repetition_penalty"] = 1.05

    gen = model.generate(**enc, **gen_kwargs)

    out_text = tok.decode(gen[0], skip_special_tokens=True)

    objs = extract_all_json_dicts(out_text)
    if not objs:
        raise RuntimeError("Model output did not contain valid JSON.\n\nOUTPUT:\n" + out_text)

    # pick family from the first dict that declares it
    family = None
    for o in objs:
        if isinstance(o.get("family"), str):
            family = o["family"]
            break

    # merge template_inputs from all dicts that have it
    template_inputs = {}
    for o in objs:
        ti = o.get("template_inputs")
        if isinstance(ti, dict):
            template_inputs.update(ti)

    # fallback: sometimes the model outputs bare dicts of params (no template_inputs wrapper)
    if not template_inputs:
        for o in objs:
            if isinstance(o, dict):
                template_inputs.update(o)

    if not isinstance(family, str) or not template_inputs:
        raise RuntimeError(
            "Could not build (family, template_inputs) from model output.\n\nOUTPUT:\n" + out_text
        )

    req = REQUIRED_KEYS.get(family, [])
    missing = [k for k in req if k not in template_inputs]

    if missing:
        # Ask the model only for missing keys (ONE retry)
        fix_prompt = (
            "Return exactly ONE JSON object and nothing else.\n"
            f'family must be "{family}".\n'
            "Output JSON only in this exact form:\n"
            '{"family":"%s","template_inputs":{...}}\n'
            f"Fill ONLY these missing keys in template_inputs: {missing}.\n\n"
            f"Original instruction:\n{args.prompt}\n"
        )
        prefix2 = "### Instruction:\n" + fix_prompt.strip() + "\n\n### Response:\n"
        enc2 = tok(prefix2, return_tensors="pt")
        enc2 = {k: v.to(device) for k, v in enc2.items()}
        gen2 = model.generate(**enc2, **gen_kwargs)
        out2 = tok.decode(gen2[0], skip_special_tokens=True)

        objs2 = extract_all_json_dicts(out2)
        for o in objs2:
            ti = o.get("template_inputs")
            if isinstance(ti, dict):
                template_inputs.update(ti)

        # recompute missing (optional)
        missing = [k for k in req if k not in template_inputs]

    pred = {"family": family, "template_inputs": template_inputs}


    rec = {
        "id": "inferred_0001",
        "prompt": args.prompt,
        "family": family,
        "template_inputs": template_inputs,
    }

    with out_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("OK")
    print("family:", family)
    print("out_jsonl:", str(out_path))


if __name__ == "__main__":
    main()
