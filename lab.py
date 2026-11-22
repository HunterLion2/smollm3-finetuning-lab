"""Reproducible SmolLM3 LoRA experiment; importing this module has no side effects."""
import argparse
import hashlib
import importlib.metadata
import json
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

def normalize(text):
    return " ".join(text.casefold().split())

def make_splits(train, test, labels, config):
    """Reserve original test first, deduplicate, then stratify train/validation."""
    rng, seen, clean_test, clean_train = random.Random(config["seed"]), set(), [], []
    removed = 0
    for split, source, target in [("test", test, clean_test), ("train", train, clean_train)]:
        for i, row in enumerate(source):
            text = normalize(row["text"])
            if text in seen:
                if split == "train":
                    removed += 1
                continue
            seen.add(text)
            target.append({"id": f"{split}:{i}", "text": row["text"], "label": labels[row["label"]]})
    out = {"train": [], "validation": [], "test": []}
    for label in labels:
        candidates = [r for r in clean_train if r["label"] == label]
        held_out = [r for r in clean_test if r["label"] == label]
        rng.shuffle(candidates)
        rng.shuffle(held_out)
        n, v, t = (config[k] for k in ["train_per_label", "validation_per_label", "test_per_label"])
        if min(n, v, t) < 1 or len(candidates) < n + v or len(held_out) < t:
            raise ValueError(f"Insufficient unique examples for {label}")
        out["train"].extend(candidates[:n])
        out["validation"].extend(candidates[n:n+v])
        out["test"].extend(held_out[:t])
    for rows in out.values():
        rng.shuffle(rows)
    return out, removed

def validate_splits(splits, labels):
    seen_text, seen_ids = set(), set()
    for split in ["train", "validation", "test"]:
        if not splits.get(split):
            raise ValueError(f"Empty split: {split}")
        for row in splits[split]:
            text = normalize(row["text"])
            if not text or row["label"] not in labels:
                raise ValueError("Empty text or unknown label")
            if text in seen_text or row["id"] in seen_ids:
                raise ValueError("Duplicate or train/evaluation leakage")
            seen_text.add(text)
            seen_ids.add(row["id"])

def load_data():
    bundle = read(ROOT / "data/experiment.json")
    if digest(bundle["splits"]) != bundle["manifest"]["data_sha256"]:
        raise ValueError("Data changed after preparation")
    validate_splits(bundle["splits"], bundle["labels"])
    return bundle

def prepare(args):
    from datasets import load_dataset
    from huggingface_hub import HfApi
    config = read(args.config)
    target = ROOT / "data/experiment.json"
    if target.exists():
        raise ValueError("Prepared snapshot exists; preserve it before starting another experiment.")
    api = HfApi()
    model_sha = api.model_info(config["model_id"], revision=config["model_revision"]).sha
    dataset_sha = api.dataset_info(config["dataset_id"], revision=config["dataset_revision"]).sha
    dataset = load_dataset(config["dataset_id"], revision=dataset_sha)
    labels = dataset["train"].features["label"].names
    splits, removed = make_splits(dataset["train"], dataset["test"], labels, config)
    validate_splits(splits, labels)
    manifest = {
        "model_id": config["model_id"], "model_revision": model_sha,
        "dataset_id": config["dataset_id"], "dataset_revision": dataset_sha,
        "data_sha256": digest(splits), "config": config,
        "excluded_train_duplicates_or_test_overlap": removed,
        "counts": {k: len(v) for k, v in splits.items()},
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    write(target, {"manifest": manifest, "labels": labels, "splits": splits})
    write(ROOT / "runs/data-manifest.json", manifest)
    print("Prepared:", manifest["counts"])

def prompt(tokenizer, text, labels):
    system = (
        "Classify the banking support request. Return exactly one label from this list, "
        "with no explanation: " + ", ".join(labels) + ". /no_think"
    )
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": text}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )

def environment():
    names = ["torch", "transformers", "trl", "peft", "datasets", "accelerate", "huggingface-hub"]
    return {name: importlib.metadata.version(name) for name in names}

def load_model(bundle):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available():
        raise RuntimeError("This recipe requires CUDA. No automatic paid job is launched.")
    m = bundle["manifest"]
    kwargs = {"revision": m["model_revision"], "trust_remote_code": False}
    tokenizer = AutoTokenizer.from_pretrained(m["model_id"], **kwargs)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        m["model_id"], torch_dtype=dtype, device_map={"": 0}, **kwargs,
    )
    return model, tokenizer, dtype

def train(args):
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import set_seed
    from trl import SFTConfig, SFTTrainer
    bundle = load_data()
    config = bundle["manifest"]["config"]
    out = ROOT / "outputs/adapter"
    if out.exists():
        raise ValueError("Adapter already exists; preserve the previous run.")
    set_seed(config["seed"])
    model, tokenizer, dtype = load_model(bundle)

    def formatted(split):
        rows = []
        for row in bundle["splits"][split]:
            p = prompt(tokenizer, row["text"], bundle["labels"])
            completion = row["label"] + tokenizer.eos_token
            if len(tokenizer(p + completion, add_special_tokens=False)["input_ids"]) > config["max_length"]:
                raise ValueError("Example exceeds max_length; do not silently truncate.")
            rows.append({"prompt": p, "completion": completion})
        return Dataset.from_list(rows)

    settings = SFTConfig(
        output_dir=str(ROOT / "outputs/checkpoints"),
        num_train_epochs=config["epochs"], learning_rate=config["learning_rate"],
        per_device_train_batch_size=config["batch_size"], per_device_eval_batch_size=1,
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        max_length=config["max_length"], completion_only_loss=True, packing=False,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=dtype == torch.bfloat16, fp16=dtype == torch.float16,
        logging_steps=5, report_to="none", push_to_hub=False, seed=config["seed"],
    )
    model.config.use_cache = False
    trainer = SFTTrainer(
        model=model, processing_class=tokenizer, args=settings,
        train_dataset=formatted("train"), eval_dataset=formatted("validation"),
        peft_config=LoraConfig(
            r=config["lora_r"], lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"], target_modules=config["target_modules"],
            task_type="CAUSAL_LM", bias="none",
        ),
    )
    output = trainer.train()
    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))
    metadata = {
        "status": "training_completed", "manifest": bundle["manifest"],
        "metrics": output.metrics, "environment": environment(),
        "gpu": torch.cuda.get_device_name(0), "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    write(out / "experiment.json", metadata)
    write(ROOT / "runs/training.json", metadata)
    (ROOT / "runs/pip-freeze.txt").write_text(
        subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True),
        encoding="utf-8",
    )
    print("Training complete. Held-out test evaluation is a separate command.")

def scores(rows, labels):
    if not rows:
        raise ValueError("Cannot score empty evaluation")
    return {
        "accuracy": sum(r["prediction"] == r["label"] for r in rows) / len(rows),
        "invalid_output_rate": sum(r["prediction"] not in labels for r in rows) / len(rows),
        "n": len(rows),
    }

def evaluate(args):
    import torch
    from peft import PeftModel
    from transformers import set_seed
    bundle = load_data()
    config = bundle["manifest"]["config"]
    output = ROOT / f"runs/{args.mode}.json"
    if output.exists():
        raise ValueError("Evaluation exists; preserve it before rerunning.")
    set_seed(config["seed"])
    adapter = ROOT / "outputs/adapter"
    if args.mode == "adapter" and read(adapter / "experiment.json")["manifest"] != bundle["manifest"]:
        raise ValueError("Adapter and evaluation provenance differ")
    model, tokenizer, _ = load_model(bundle)
    if args.mode == "adapter":
        model = PeftModel.from_pretrained(model, str(adapter))
    model.eval()
    rows = []
    for row in bundle["splits"]["test"]:
        inputs = tokenizer(
            prompt(tokenizer, row["text"], bundle["labels"]),
            return_tensors="pt", add_special_tokens=False,
        ).to(model.device)
        if inputs["input_ids"].shape[-1] + config["max_new_tokens"] > config["max_length"]:
            raise ValueError("Evaluation exceeds configured context; do not silently truncate.")
        with torch.inference_mode():
            generated = model.generate(
                **inputs, max_new_tokens=config["max_new_tokens"], do_sample=False,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            )
        raw = tokenizer.decode(
            generated[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True,
        )
        rows.append({**row, "raw_output": raw, "prediction": raw.strip()})
    write(output, {
        "status": "evaluated", "mode": args.mode, "manifest": bundle["manifest"],
        "environment": environment(),
        "generation": {"do_sample": False, "max_new_tokens": config["max_new_tokens"]},
        "metrics": scores(rows, bundle["labels"]), "rows": rows,
    })
    print(output)

def compare_runs(base, tuned):
    if base.get("status") != "evaluated" or tuned.get("status") != "evaluated":
        raise ValueError("Both real evaluation runs are required")
    if base.get("mode") != "base" or tuned.get("mode") != "adapter":
        raise ValueError("Expected base and adapter runs")
    if base["manifest"] != tuned["manifest"] or base["generation"] != tuned["generation"]:
        raise ValueError("Different dataset, model or generation settings")
    if base.get("environment") != tuned.get("environment"):
        raise ValueError("Different runtime dependencies")
    a, b = base["rows"], tuned["rows"]
    identity = lambda rows: [(r["id"], r["text"], r["label"]) for r in rows]
    if identity(a) != identity(b):
        raise ValueError("Evaluation examples differ")
    labels = sorted(set(r["label"] for r in a))
    before, after = scores(a, labels), scores(b, labels)
    return before, after, after["accuracy"] - before["accuracy"]

def compare(args):
    base, tuned = read(ROOT / "runs/base.json"), read(ROOT / "runs/adapter.json")
    before, after, delta = compare_runs(base, tuned)
    report = [
        "# Gerçek değerlendirme sonuçları", "",
        "| Ölçüt | Temel model | LoRA |", "| --- | ---: | ---: |",
        f"| Doğruluk | {before['accuracy']:.4f} | {after['accuracy']:.4f} |",
        f"| Geçersiz çıktı | {before['invalid_output_rate']:.4f} | {after['invalid_output_rate']:.4f} |",
        "", f"Örnek sayısı: {before['n']}. Doğruluk farkı: {delta:+.4f}.",
        "İyileşme gözlendi." if delta > 0 else "İyileşme gözlenmedi (eşit veya daha kötü).",
        "", "Tek seed ve küçük görev kümesi; genel LLM veya Türkçe/klinik kalite sonucu değildir.",
        "", "## İlk 10 sabit test örneği (başarıya göre seçilmedi)", "",
    ]
    for a, b in zip(base["rows"][:10], tuned["rows"][:10]):
        report.extend([
            "    " + json.dumps({
                "id": a["id"], "text": a["text"], "expected": a["label"],
                "base": a["raw_output"], "adapter": b["raw_output"],
            }, ensure_ascii=False), "",
        ])
    report.extend([
        "## Hata analizi", "",
        "- Tüm hataları runs/*.json içinde incele; hangi sınıflar karışıyor?",
        "- Veri belirsizliği, geçersiz çıktılar ve kötüleşen örnekleri kaydet.",
        "- Küçük örneklem ve tek seed sınırlamalarını açıkla.",
        "- Test kümesine tekrar tekrar parametre ayarlama; yeni deneyde ayrı doğrulama planı kullan.",
    ])
    (ROOT / "runs/comparison.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(ROOT / "runs/comparison.md")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--config", type=Path, default=ROOT / "config.json")
    p.set_defaults(func=prepare)
    commands.add_parser("train").set_defaults(func=train)
    p = commands.add_parser("evaluate")
    p.add_argument("--mode", choices=["base", "adapter"], required=True)
    p.set_defaults(func=evaluate)
    commands.add_parser("compare").set_defaults(func=compare)
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
