"""在独立泊车弱监督数据上微调 Qwen3-VL 语言骨干的 LoRA adapter。"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

from parksight_vlm.assessment import ParkingAssessment, ParkingRiskEvent
from parksight_vlm.workload import FrozenWorkload


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


def _training_event_coverage(
    train_records: list[dict[str, Any]],
    validation_records: list[dict[str, Any]],
    recommended_min_train_support: int = 3,
) -> dict[str, Any]:
    """Return event coverage warnings for the cheap pre-training validation."""
    if recommended_min_train_support < 1:
        raise ValueError("recommended_min_train_support must be at least 1")

    def counts(records: list[dict[str, Any]]) -> dict[str, int]:
        observed = Counter(
            event
            for record in records
            for event in record.get("assessment", {}).get("events", [])
        )
        return {event.value: observed.get(event.value, 0) for event in ParkingRiskEvent}

    train_counts = counts(train_records)
    validation_counts = counts(validation_records)
    warnings: list[dict[str, Any]] = []
    for event in ParkingRiskEvent:
        event_name = event.value
        train_count = train_counts[event_name]
        if train_count == 0:
            warnings.append({"type": "missing_from_train", "event": event_name})
        elif train_count < recommended_min_train_support:
            warnings.append(
                {
                    "type": "low_train_support",
                    "event": event_name,
                    "count": train_count,
                    "recommended_minimum": recommended_min_train_support,
                }
            )
        if validation_counts[event_name] > 0 and train_count == 0:
            warnings.append(
                {"type": "validation_not_seen_in_train", "event": event_name}
            )
    return {
        "train": train_counts,
        "validation": validation_counts,
        "recommended_min_train_support": recommended_min_train_support,
        "warnings": warnings,
    }


def enforce_training_event_coverage(
    coverage: dict[str, Any],
    *,
    min_train_event_support: int = 3,
    require_validation_event_coverage: bool = True,
) -> None:
    """在训练前将事件覆盖要求提升为硬门禁。"""
    if min_train_event_support < 1:
        raise ValueError("min_train_event_support must be at least 1")
    violations: list[str] = []
    for event in (event.value for event in ParkingRiskEvent):
        count = coverage["train"].get(event, 0)
        if count < min_train_event_support:
            violations.append(
                f"train:{event}={count}<{min_train_event_support}"
            )
        if require_validation_event_coverage and coverage["validation"].get(event, 0) == 0:
            violations.append(f"validation:{event}=0")
    if violations:
        raise ValueError("event coverage gate failed: " + ", ".join(violations))


def validate_training_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate all cheap training inputs without importing CUDA dependencies."""
    dataset_path = Path(config["dataset_path"]).resolve()
    workload_path = Path(config["workload_path"]).resolve()
    model_path = Path(config["model_path"]).resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"training dataset does not exist: {dataset_path}")
    if not workload_path.is_file():
        raise FileNotFoundError(f"workload does not exist: {workload_path}")
    if not model_path.is_dir():
        raise FileNotFoundError(f"model directory does not exist: {model_path}")

    workload = FrozenWorkload.load(workload_path)
    records = _load_records(dataset_path)
    label_source = _validate_label_provenance(records, config)
    factor = int(config.get("non_low_oversampling_factor", 1))
    event_factor = int(config.get("event_oversampling_factor", 1))
    unique_train_records = [record for record in records if record.get("split") == "train"]
    validation_records = [
        record for record in records if record.get("split") == "validation"
    ]
    if not unique_train_records or not validation_records:
        raise ValueError("dataset requires non-empty train and validation splits")
    for index, record in enumerate(records, start=1):
        image = record.get("image")
        if not isinstance(image, str) or not Path(image).is_file():
            raise FileNotFoundError(f"training image is missing at row {index}: {image}")
        ParkingAssessment.from_mapping(record.get("assessment"))

    event_coverage = _training_event_coverage(
        unique_train_records,
        validation_records,
    )
    require_event_coverage = config.get("require_event_coverage", False)
    if not isinstance(require_event_coverage, bool):
        raise ValueError("require_event_coverage must be a boolean")
    if require_event_coverage:
        enforce_training_event_coverage(
            event_coverage,
            min_train_event_support=int(config.get("min_train_event_support", 3)),
            require_validation_event_coverage=bool(
                config.get("require_validation_event_coverage", True)
            ),
        )
    effective_train_records = _oversample_training_records(
        unique_train_records, factor, event_factor
    )
    return {
        "status": "validated",
        "dataset_path": str(dataset_path),
        "workload_path": str(workload_path),
        "workload_identity": workload.identity,
        "model_path": str(model_path),
        "label_source": label_source,
        "sample_count": len(records),
        "unique_train_samples": len(unique_train_records),
        "effective_train_samples": len(effective_train_records),
        "non_low_oversampling_factor": factor,
        "event_oversampling_factor": event_factor,
        "validation_samples": len(validation_records),
        "event_coverage": event_coverage,
        "require_event_coverage": require_event_coverage,
    }


def _validate_label_provenance(
    records: list[dict[str, Any]], config: dict[str, Any]
) -> str:
    """Prevent candidate labels from entering a formal training run by accident."""
    configured_source = config.get("label_source")
    if not isinstance(configured_source, str) or not configured_source.strip():
        raise ValueError(
            "training config must declare a non-blank label_source; "
            "use human_confirmed_v1 after review finalization"
        )
    configured_source = configured_source.strip()
    observed_sources = {
        record.get("label_source")
        for record in records
        if isinstance(record.get("label_source"), str)
    }
    if len(observed_sources) != 1 or configured_source not in observed_sources:
        raise ValueError(
            "dataset label_source does not match training config: "
            f"configured={configured_source!r}, observed={sorted(observed_sources)!r}"
        )
    if (
        configured_source == "codex_visual_review_v1_single_pass"
        and not bool(config.get("allow_candidate_labels", False))
    ):
        raise ValueError(
            "candidate Codex labels are blocked for formal LoRA training; "
            "finalize the review package and use label_source='human_confirmed_v1'"
        )
    return configured_source


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


def _oversample_training_records(
    records: list[dict[str, Any]],
    non_low_factor: int,
    event_factor: int,
) -> list[dict[str, Any]]:
    """Increase event exposure while retaining the non-low safety weighting."""
    if non_low_factor < 1:
        raise ValueError("non_low_oversampling_factor must be at least 1")
    if event_factor < 1:
        raise ValueError("event_oversampling_factor must be at least 1")
    expanded: list[dict[str, Any]] = []
    for record in records:
        assessment = record.get("assessment", {})
        risk_level = assessment.get("risk_level")
        has_event = bool(assessment.get("events"))
        repeats = non_low_factor if risk_level != "low" else 1
        if has_event:
            repeats = max(repeats, event_factor)
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
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate training inputs and provenance without loading CUDA or a model",
    )
    args = parser.parse_args()
    config = _load_json(args.config)

    validation = validate_training_config(config)
    if args.validate_only:
        print(json.dumps(validation, ensure_ascii=False, indent=2))
        return 0

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoProcessor,
        Qwen3VLForConditionalGeneration,
        get_linear_schedule_with_warmup,
    )

    seed = int(config["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("LoRA training requires CUDA")

    dataset_path = Path(config["dataset_path"]).resolve()
    workload = FrozenWorkload.load(Path(config["workload_path"]))
    records = _load_records(dataset_path)
    label_source = _validate_label_provenance(records, config)
    unique_train_records = [
        record for record in records if record["split"] == "train"
    ]
    train_records = _oversample_training_records(
        unique_train_records,
        int(config.get("non_low_oversampling_factor", 1)),
        int(config.get("event_oversampling_factor", 1)),
    )
    event_factor = int(config.get("event_oversampling_factor", 1))
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
        "label_source": label_source,
        "workload_identity": workload.identity,
        "train_samples": len(train_records),
        "unique_train_samples": len(unique_train_records),
        "non_low_oversampling_factor": int(
            config.get("non_low_oversampling_factor", 1)
        ),
        "event_oversampling_factor": event_factor,
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
