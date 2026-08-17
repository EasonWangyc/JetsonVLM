"""在独立泊车弱监督数据上微调 Qwen3-VL 语言骨干的 LoRA adapter。"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("training config must be a JSON object")
    return payload


def _load_records(path: Path) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not records:
        raise ValueError("training dataset must not be empty")
    return records


def _oversample_non_low_records(
    records: list[dict[str, Any]], factor: int
) -> list[dict[str, Any]]:
    if factor < 1:
        raise ValueError("non_low_oversampling_factor must be at least 1")
    expanded: list[dict[str, Any]] = []
    for record in records:
        risk_level = record.get("assessment", {}).get("risk_level")
        repeats = factor if risk_level != "low" else 1
        expanded.extend([record] * repeats)
    return expanded


def _messages(image: Any, workload: Any, answer: str | None = None) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": [{"type": "text", "text": workload.system_prompt}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": workload.render_user_prompt()},
            ],
        },
    ]
    if answer is not None:
        messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": answer}]}
        )
    return messages


def _prepare_sample(processor: Any, workload: Any, record: dict[str, Any], device: Any) -> dict[str, Any]:
    from PIL import Image

    with Image.open(record["image"]) as source:
        image = source.convert("RGB").resize(
            (workload.input_size.width, workload.input_size.height)
        )
    answer = json.dumps(record["assessment"], ensure_ascii=False, separators=(",", ":"))
    prompt = processor.apply_chat_template(
        _messages(image, workload),
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    full = processor.apply_chat_template(
        _messages(image, workload, answer),
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_tensors="pt",
    )
    full.pop("token_type_ids", None)
    labels = full["input_ids"].clone()
    prompt_tokens = min(prompt["input_ids"].shape[1], labels.shape[1])
    labels[:, :prompt_tokens] = -100
    full["labels"] = labels
    return {key: value.to(device) for key, value in full.items()}


def _evaluate(model: Any, processor: Any, workload: Any, records: list[dict[str, Any]], device: Any) -> float:
    import torch

    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for record in records:
            batch = _prepare_sample(processor, workload, record, device)
            losses.append(float(model(**batch).loss.detach().cpu()))
    model.train()
    return sum(losses) / len(losses)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = _load_json(args.config)

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoProcessor,
        Qwen3VLForConditionalGeneration,
        get_linear_schedule_with_warmup,
    )

    from parksight_vlm.workload import FrozenWorkload

    seed = int(config["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("LoRA training requires CUDA")

    dataset_path = Path(config["dataset_path"]).resolve()
    workload = FrozenWorkload.load(Path(config["workload_path"]))
    records = _load_records(dataset_path)
    unique_train_records = [
        record for record in records if record["split"] == "train"
    ]
    train_records = _oversample_non_low_records(
        unique_train_records,
        int(config.get("non_low_oversampling_factor", 1)),
    )
    validation_records = [
        record for record in records if record["split"] == "validation"
    ]
    if not train_records or not validation_records:
        raise ValueError("dataset requires non-empty train and validation splits")

    model_path = config["model_path"]
    processor = AutoProcessor.from_pretrained(model_path)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation=config.get("attn_implementation", "sdpa"),
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    lora_config = LoraConfig(
        r=int(config["lora_rank"]),
        lora_alpha=int(config["lora_alpha"]),
        lora_dropout=float(config["lora_dropout"]),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(config["target_modules"]),
    )
    model = get_peft_model(model, lora_config)
    unexpected = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "language_model" not in name
    ]
    if unexpected:
        raise RuntimeError(f"LoRA unexpectedly targets non-language modules: {unexpected[:5]}")
    model.to("cuda")

    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(config["learning_rate"]),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )
    epochs = int(config["epochs"])
    accumulation = int(config["gradient_accumulation_steps"])
    optimizer_steps = math.ceil(len(train_records) / accumulation) * epochs
    warmup_steps = int(optimizer_steps * float(config.get("warmup_ratio", 0.0)))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=optimizer_steps
    )

    output_dir = Path(config["output_directory"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "training_metrics.jsonl"
    optimizer.zero_grad(set_to_none=True)
    step = 0
    started = time.time()
    peak_memory = 0.0
    model.train()
    with metrics_path.open("w", encoding="utf-8", newline="\n") as metrics:
        for epoch in range(epochs):
            random.Random(seed + epoch).shuffle(train_records)
            for sample_index, record in enumerate(train_records, start=1):
                batch = _prepare_sample(processor, workload, record, model.device)
                loss = model(**batch).loss / accumulation
                loss.backward()
                if sample_index % accumulation == 0 or sample_index == len(train_records):
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], 1.0
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    step += 1
                    peak_memory = max(
                        peak_memory,
                        torch.cuda.max_memory_allocated() / (1024**3),
                    )
                    item = {
                        "optimizer_step": step,
                        "epoch": epoch + 1,
                        "loss": float(loss.detach().cpu()) * accumulation,
                        "learning_rate": scheduler.get_last_lr()[0],
                    }
                    metrics.write(json.dumps(item, ensure_ascii=False) + "\n")
                    metrics.flush()
                    print(json.dumps(item, ensure_ascii=False), flush=True)

    validation_loss = _evaluate(
        model, processor, workload, validation_records, model.device
    )
    model.save_pretrained(output_dir, safe_serialization=True)
    processor.save_pretrained(output_dir)
    summary = {
        "status": "succeeded",
        "base_model": model_path,
        "model_revision": config["model_revision"],
        "dataset_path": str(dataset_path),
        "workload_identity": workload.identity,
        "train_samples": len(train_records),
        "unique_train_samples": len(unique_train_records),
        "non_low_oversampling_factor": int(
            config.get("non_low_oversampling_factor", 1)
        ),
        "validation_samples": len(validation_records),
        "epochs": epochs,
        "optimizer_steps": optimizer_steps,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_ratio": trainable / total,
        "validation_loss": validation_loss,
        "peak_cuda_memory_gib": peak_memory,
        "elapsed_seconds": time.time() - started,
        "precision": "bfloat16",
        "target_modules": list(config["target_modules"]),
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
