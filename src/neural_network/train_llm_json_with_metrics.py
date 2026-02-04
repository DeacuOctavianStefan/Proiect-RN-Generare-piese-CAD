import argparse
import json
import random
import inspect
import time
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

# --- Optional deps (LoRA) ---
try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False

# --- HuggingFace core ---
try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
        set_seed,
    )
except Exception as e:
    raise RuntimeError(
        "Missing transformers. Install with:\n"
        "  pip install transformers datasets accelerate\n"
        "Optional (LoRA):\n"
        "  pip install peft\n"
    ) from e

try:
    from datasets import Dataset
except Exception as e:
    raise RuntimeError(
        "Missing datasets. Install with:\n"
        "  pip install datasets\n"
    ) from e


# -----------------------------
# Data helpers
# -----------------------------
def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def stable_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def build_target_row(rec: Dict[str, Any]) -> Dict[str, Any]:
    family = rec.get("family")
    inputs = rec.get("template_inputs") or {}
    return {"family": family, "template_inputs": inputs}


def format_example(prompt: str, target_obj: Dict[str, Any]) -> str:
    return (
        "### Instruction:\n"
        f"{prompt.strip()}\n\n"
        "### Response:\n"
        f"{stable_json(target_obj)}"
    )


def prepare_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rows:
        prompt = r.get("prompt")
        if not prompt:
            continue
        target = build_target_row(r)
        text = format_example(prompt, target)
        out.append({"text": text, "target": target})
    return out


# -----------------------------
# Metrics helpers
# -----------------------------
def try_parse_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    snippet = text[start:end + 1]
    try:
        return json.loads(snippet)
    except Exception:
        return None


def flatten_numeric(d: Dict[str, Any], prefix: str = "") -> Dict[str, float]:
    flat: Dict[str, float] = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            flat.update(flatten_numeric(v, key))
        else:
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                flat[key] = float(v)
    return flat



# Required keys per family (used for accuracy-style metrics)
REQUIRED_KEYS: Dict[str, List[str]] = {
    "shaft": ["SHAFT_OD", "SHAFT_L", "BORE_D", "CHAMFER_L", "CHAMFER_ANG"],
    "bearing_sleeve": ["SLEEVE_OD", "SLEEVE_L", "BORE_D", "CHAMFER_L", "CHAMFER_ANG"],
    "bearing_flanged": ["FLANGE_OD", "FLANGE_THK", "BORE_D", "BOLT_D", "BOLT_PCD", "BOLT_COUNT", "BOLT_DEPTH"],
}


def _percentile(xs: List[float], p: float) -> float:
    if not xs:
        return float("nan")
    xs2 = sorted(xs)
    k = int(round((p / 100.0) * (len(xs2) - 1)))
    k = max(0, min(k, len(xs2) - 1))
    return xs2[k]


def _macro_f1(y_true: List[str], y_pred: List[str]) -> float:
    # Macro-F1 over labels present in y_true (common for imbalanced datasets).
    labels = sorted(set(y_true))
    if not labels:
        return float("nan")

    # confusion components per label
    f1s: List[float] = []
    for lab in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p == lab)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != lab and p == lab)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p != lab)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return float(sum(f1s) / len(f1s))


def _within_tol(pred_v: Any, gold_v: Any, abs_tol: float, rel_tol: float) -> bool:
    try:
        pv = float(pred_v)
        gv = float(gold_v)
    except Exception:
        return False
    diff = abs(pv - gv)
    return diff <= abs_tol + rel_tol * abs(gv)

from transformers import TrainerCallback

class GenMetricsCallback(TrainerCallback):
    def __init__(self, compute_metrics_fn):
        self.compute_metrics_fn = compute_metrics_fn

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        trainer = kwargs.get("trainer", None)
        if trainer is None:
            return
        # Run your custom generation-based metrics
        extra = self.compute_metrics_fn()
        trainer.log(extra)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", required=True)
    ap.add_argument("--train_jsonl", default="data/processed/train.jsonl")
    ap.add_argument("--val_jsonl", default="data/processed/validation.jsonl")
    ap.add_argument("--test_jsonl", default="data/processed/test.jsonl")
    ap.add_argument("--max_steps", type=int, default=-1, help="If >0, train only this many steps.")
    ap.add_argument("--resume_from_checkpoint", default="", help="Path to checkpoint folder to resume.")

    ap.add_argument("--model_name", default="gpt2")
    ap.add_argument("--run_name", default="json_sft")
    ap.add_argument("--output_dir", default="runs")

    ap.add_argument("--max_length", type=int, default=384)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--train_batch_size", type=int, default=1)
    ap.add_argument("--eval_batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)

    # LoRA (optional)
    ap.add_argument("--use_lora", type=int, default=0)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)

    # precision (GPU only)
    ap.add_argument("--bf16", type=int, default=0)
    ap.add_argument("--fp16", type=int, default=0)

    # eval/generation settings for metric probes
    ap.add_argument("--eval_sample_size", type=int, default=64, help="How many validation rows to sample for accuracy/F1/latency metrics each eval.")
    ap.add_argument("--eval_max_new_tokens", type=int, default=160, help="max_new_tokens used during eval generation probes.")
    ap.add_argument("--eval_abs_tol", type=float, default=0.0, help="Absolute tolerance for numeric accuracy checks.")
    ap.add_argument("--eval_rel_tol", type=float, default=0.0, help="Relative tolerance for numeric accuracy checks (e.g., 0.01 = 1%).")

    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    train_path = (repo / args.train_jsonl).resolve()
    val_path = (repo / args.val_jsonl).resolve()
    test_path = (repo / args.test_jsonl).resolve()

    if not train_path.exists():
        raise FileNotFoundError(f"train_jsonl not found: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"val_jsonl not found: {val_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"test_jsonl not found: {test_path}")

    out_root = (repo / args.output_dir / args.run_name).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)

    # Load data
    train_rows = prepare_rows(read_jsonl(train_path))
    val_rows = prepare_rows(read_jsonl(val_path))
    test_rows = prepare_rows(read_jsonl(test_path))

    print(f"[data] train rows: {len(train_rows)}")
    print(f"[data] val rows:   {len(val_rows)}")
    print(f"[data] test rows:  {len(test_rows)}")

    # Tokenizer / model
    tok = AutoTokenizer.from_pretrained(args.model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name)

    # Optional LoRA
    if args.use_lora and PEFT_AVAILABLE:
        print("[lora] enabled")
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=None,
        )
        model = get_peft_model(model, lora_cfg)
        try:
            model.print_trainable_parameters()
        except Exception:
            pass
    elif args.use_lora and not PEFT_AVAILABLE:
        print("[lora] requested but peft not installed -> continuing without LoRA")

    # Tokenize
    def tok_fn(batch: Dict[str, List[Any]]) -> Dict[str, Any]:
        return tok(
            batch["text"],
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
        )

    ds_train_raw = Dataset.from_list(train_rows)
    ds_val_raw = Dataset.from_list(val_rows)
    ds_test_raw = Dataset.from_list(test_rows)

    ds_train = ds_train_raw.map(
        tok_fn,
        batched=True,
        remove_columns=ds_train_raw.column_names,  # removes text/target so no strings remain
    )
    ds_val = ds_val_raw.map(
        tok_fn,
        batched=True,
        remove_columns=ds_val_raw.column_names,
    )
    ds_test = ds_test_raw.map(
        tok_fn,
        batched=True,
        remove_columns=ds_test_raw.column_names,
    )


    # Causal LM labels = input_ids
    def add_labels(ex):
        ex["labels"] = ex["input_ids"].copy()
        return ex

    ds_train = ds_train.map(add_labels)
    ds_val = ds_val.map(add_labels)
    ds_test = ds_test.map(add_labels)

    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    use_cuda = torch.cuda.is_available()
    print(f"[device] cuda={use_cuda}")

    # -----------------------------
    # TrainingArguments: compatibility across transformers versions
    # -----------------------------
    sig = inspect.signature(TrainingArguments.__init__)
    has_eval_strategy = "evaluation_strategy" in sig.parameters
    has_eval_strategy_new = "eval_strategy" in sig.parameters
    has_save_strategy = "save_strategy" in sig.parameters

    ta_kwargs = dict(
        output_dir=str(out_root),
        run_name=args.run_name,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        eval_steps=200,
        save_steps=200,
        save_total_limit=3,
        logging_steps=50,
        report_to="none",
        remove_unused_columns=True,
        save_strategy="steps",
    )
    # Prevent eval from storing logits/preds (saves a TON of VRAM)
    ta_kwargs["prediction_loss_only"] = True
    ta_kwargs["eval_accumulation_steps"] = 1
    ta_kwargs["dataloader_pin_memory"] = use_cuda  # optional; harmless
    
    if args.max_steps and args.max_steps > 0:
        ta_kwargs["max_steps"] = args.max_steps
    
    # precision only if GPU
    ta_kwargs["fp16"] = bool(args.fp16) if use_cuda else False
    ta_kwargs["bf16"] = bool(args.bf16) if use_cuda else False

    if has_eval_strategy:
        ta_kwargs["evaluation_strategy"] = "steps"
    elif has_eval_strategy_new:
        ta_kwargs["eval_strategy"] = "steps"
    else:
        print("[warn] TrainingArguments has no evaluation strategy arg; disabling periodic evaluation.")

    if has_save_strategy:
        ta_kwargs["save_strategy"] = "steps"

    targs = TrainingArguments(**ta_kwargs)

    # -----------------------------
    # Custom eval: JSON parse rate + MAE on numeric fields (lightweight)
    # -----------------------------
    def compute_metrics(eval_pred):
        # Lightweight metric probes via greedy generation on a small validation sample.
        n = min(args.eval_sample_size, len(val_rows))
        idxs = random.sample(range(len(val_rows)), k=n) if len(val_rows) >= n else list(range(len(val_rows)))

        parse_ok = 0
        family_ok = 0
        keys_ok = 0
        sample_ok = 0

        y_true: List[str] = []
        y_pred: List[str] = []

        mae_sum: Dict[str, float] = {}
        mae_cnt: Dict[str, int] = {}

        latencies_ms: List[float] = []

        model.eval()
        with torch.no_grad():
            for i in idxs:
                text = val_rows[i]["text"]
                prefix = text.split("### Response:\n")[0] + "### Response:\n"
                enc = tok(prefix, return_tensors="pt", truncation=True, max_length=args.max_length)
                enc = {k: v.to(model.device) for k, v in enc.items()}

                # Measure generation latency (model-only) for this sample
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                gen = model.generate(
                    **enc,
                    max_new_tokens=args.eval_max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=tok.eos_token_id,
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                latencies_ms.append(1000.0 * (t1 - t0))

                out = tok.decode(gen[0], skip_special_tokens=True)

                pred = try_parse_json_from_text(out)
                gold = val_rows[i]["target"]
                if pred is None or not isinstance(gold, dict):
                    gf = gold.get("family") if isinstance(gold, dict) else None
                    if isinstance(gf, str):
                        y_true.append(gf)
                        y_pred.append("__PARSE_FAIL__")
                    continue

                parse_ok += 1

                gold_family = gold.get("family")
                pred_family = pred.get("family") if isinstance(pred, dict) else None
                if isinstance(gold_family, str):
                    y_true.append(gold_family)
                    y_pred.append(pred_family if isinstance(pred_family, str) else "__NO_FAMILY__")

                if isinstance(gold_family, str) and isinstance(pred_family, str) and pred_family == gold_family:
                    family_ok += 1

                pred_in = pred.get("template_inputs") if isinstance(pred, dict) else None
                gold_in = gold.get("template_inputs") if isinstance(gold, dict) else None

                # Accuracy-style checks: required keys present and values match (with tolerance for numerics)
                req_keys = REQUIRED_KEYS.get(gold_family, [])
                if not req_keys and isinstance(gold_in, dict):
                    req_keys = list(gold_in.keys())

                if isinstance(pred_in, dict) and isinstance(gold_in, dict) and req_keys:
                    if all(k2 in pred_in for k2 in req_keys):
                        keys_ok += 1

                        ok = True
                        for k2 in req_keys:
                            gv = gold_in.get(k2)
                            pv = pred_in.get(k2)

                            # Numeric compare with tolerance; otherwise exact match
                            if isinstance(gv, (int, float)) and not isinstance(gv, bool):
                                if pv is None:
                                    ok = False
                                    break
                                if not _within_tol(pv, gv, args.eval_abs_tol, args.eval_rel_tol):
                                    ok = False
                                    break
                            else:
                                if pv != gv:
                                    ok = False
                                    break

                        if ok:
                            sample_ok += 1

                    # Keep your existing numeric MAE (over overlapping numeric keys)
                    pflat = flatten_numeric(pred_in)
                    gflat = flatten_numeric(gold_in)
                    for k2, gv in gflat.items():
                        pv = pflat.get(k2)
                        if pv is None:
                            continue
                        err = abs(pv - gv)
                        mae_sum[k2] = mae_sum.get(k2, 0.0) + err
                        mae_cnt[k2] = mae_cnt.get(k2, 0) + 1

        denom = max(1, len(idxs))
        parse_rate = parse_ok / denom
        family_acc = family_ok / denom
        keys_acc = keys_ok / denom
        sample_acc = sample_ok / denom

        macro_f1 = _macro_f1(y_true, y_pred) if y_true else float("nan")

        if mae_cnt:
            overall = sum(mae_sum.values()) / max(1, sum(mae_cnt.values()))
        else:
            overall = float("nan")

        avg_lat_ms = float(statistics.mean(latencies_ms)) if latencies_ms else float("nan")
        p50_lat_ms = _percentile(latencies_ms, 50) if latencies_ms else float("nan")
        p90_lat_ms = _percentile(latencies_ms, 90) if latencies_ms else float("nan")

        metrics = {
            # Existing metrics
            "json_parse_rate": parse_rate,
            "mae_overall": overall,
            # Added KPIs
            "family_accuracy": family_acc,
            "required_keys_accuracy": keys_acc,
            "sample_accuracy": sample_acc,
            "f1_macro_family": macro_f1,
            # Inference latency probes (generation-only)
            "latency_ms_avg": avg_lat_ms,
            "latency_ms_p50": p50_lat_ms,
            "latency_ms_p90": p90_lat_ms,
        }
        return metrics

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        tokenizer=tok,
        data_collator=collator,
        callbacks=[GenMetricsCallback(compute_metrics)],
    )


    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint or None)

    # Save final
    print("[save] saving final model...")
    final_dir = out_root / "final"
    trainer.save_model(str(final_dir))
    tok.save_pretrained(str(final_dir))

    # Quick test parse rate
    print("[eval] running quick test generation eval...")
    n = min(128, len(test_rows))
    idxs = random.sample(range(len(test_rows)), k=n) if len(test_rows) >= n else list(range(len(test_rows)))

    parse_ok = 0
    model.eval()
    with torch.no_grad():
        for i in idxs:
            text = test_rows[i]["text"]
            prefix = text.split("### Response:\n")[0] + "### Response:\n"
            enc = tok(prefix, return_tensors="pt", truncation=True, max_length=args.max_length)
            enc = {k: v.to(model.device) for k, v in enc.items()}
            gen = model.generate(
                **enc,
                max_new_tokens=160,
                do_sample=False,
                num_beams=1,
                pad_token_id=tok.eos_token_id,
            )
            out = tok.decode(gen[0], skip_special_tokens=True)
            pred = try_parse_json_from_text(out)
            if pred is not None:
                parse_ok += 1

    print(f"[test] json_parse_rate={parse_ok / max(1, len(idxs)):.3f} over n={len(idxs)}")
    print(f"[done] model saved to: {final_dir}")


if __name__ == "__main__":
    main()
