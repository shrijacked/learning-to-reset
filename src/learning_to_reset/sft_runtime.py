"""Supervised fine-tuning runtime for prepared reset-aware datasets."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from learning_to_reset.model_resolver import resolve_model_name_or_path
from learning_to_reset.prompts import PromptExample
from learning_to_reset.sft_schema import iter_prompt_example_payloads


def load_prepared_examples(
    path: str | Path,
    max_examples: int | None = None,
    strict_input_schema: bool = True,
) -> tuple[PromptExample, ...]:
    """Load prompt/response examples from a prepared JSONL file (streaming).

    When ``max_examples`` is set, stop after that many non-blank rows so large
    files are not fully read or parsed. Strict schema mode rejects malformed
    rows before they can silently enter SFT.
    """

    if max_examples is not None and max_examples < 1:
        raise ValueError("max_examples must be >= 1 when provided")

    source = Path(path)
    examples: list[PromptExample] = []
    for _, payload in iter_prompt_example_payloads(source, strict=strict_input_schema):
        examples.append(
            PromptExample(
                prompt=payload["prompt"],
                response=payload.get("response", ""),
                metadata=dict(payload.get("metadata", {})),
            )
        )
        if max_examples is not None and len(examples) >= max_examples:
            break
    return tuple(examples)


def render_training_text(example: PromptExample) -> str:
    """Render one prompt/response pair into plain causal-LM training text."""

    prompt = example.prompt.strip()
    response = example.response.strip()
    if response:
        return f"{prompt}\n\n{response}"
    return prompt


def tokenize_training_example(
    example: PromptExample,
    tokenizer: Any,
    *,
    max_length: int,
) -> Dict[str, List[int]]:
    """Tokenize one training example and mask prompt tokens in the labels."""

    prompt_ids = list(tokenizer.encode(example.prompt.strip(), add_special_tokens=False))
    response_ids = list(tokenizer.encode(example.response.strip(), add_special_tokens=False))
    if response_ids and getattr(tokenizer, "eos_token_id", None) is not None:
        response_ids = response_ids + [int(tokenizer.eos_token_id)]

    input_ids = (prompt_ids + response_ids)[:max_length]
    attention_mask = [1] * len(input_ids)

    prompt_cutoff = min(len(prompt_ids), len(input_ids))
    labels = [-100] * prompt_cutoff + input_ids[prompt_cutoff:]
    if len(labels) < len(input_ids):
        labels.extend(input_ids[len(labels) :])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


@dataclass(frozen=True)
class PromptCollator:
    """Pad tokenized prompt/response examples into uniform batches."""

    pad_token_id: int

    def __call__(self, features: Sequence[Dict[str, List[int]]]) -> Dict[str, List[List[int]]]:
        max_length = max(len(feature["input_ids"]) for feature in features)

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        for feature in features:
            pad = max_length - len(feature["input_ids"])
            batch_input_ids.append(feature["input_ids"] + [self.pad_token_id] * pad)
            batch_attention_mask.append(feature["attention_mask"] + [0] * pad)
            batch_labels.append(feature["labels"] + [-100] * pad)

        batch = {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "labels": batch_labels,
        }
        try:
            import torch
        except Exception:
            return batch
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def _require_transformers():
    try:
        from datasets import Dataset
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
    except Exception as exc:  # pragma: no cover - exercised only in runtime environments
        raise RuntimeError(
            "transformers/datasets are required for SFT training. "
            "Install them in the active environment first."
        ) from exc
    return Dataset, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


def _build_dataset(examples: Sequence[PromptExample], tokenizer: Any, max_length: int):
    Dataset, _, _, _, _ = _require_transformers()
    rows = [tokenize_training_example(example, tokenizer, max_length=max_length) for example in examples]
    return Dataset.from_list(rows)


def train_sft(
    *,
    train_path: str | Path,
    output_dir: str | Path,
    model_name_or_path: str,
    validation_path: str | Path | None = None,
    max_length: int = 1024,
    learning_rate: float = 2e-5,
    num_train_epochs: float = 1.0,
    train_batch_size: int = 1,
    eval_batch_size: int = 1,
    max_train_examples: int | None = None,
    strict_input_schema: bool = True,
) -> Dict[str, Any]:
    """Run a supervised fine-tuning loop over prepared JSONL artifacts."""

    Dataset, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments = _require_transformers()
    import torch

    force_cpu = os.environ.get("LTR_FORCE_CPU", "").strip().lower() in ("1", "true", "yes")
    has_accel = torch.cuda.is_available() or (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    )
    use_cpu = force_cpu or not has_accel

    resolved_model_path = resolve_model_name_or_path(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(resolved_model_path)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})

    if max_train_examples is not None and max_train_examples < 1:
        raise ValueError("max_train_examples must be >= 1")
    train_examples = load_prepared_examples(
        train_path,
        max_train_examples,
        strict_input_schema=strict_input_schema,
    )
    if max_train_examples is not None:
        print(
            f"[sft_runtime] using first {len(train_examples)} train rows (--max-train-examples)",
            flush=True,
        )
    validation_examples = (
        load_prepared_examples(
            validation_path,
            strict_input_schema=strict_input_schema,
        )
        if validation_path
        else ()
    )

    train_dataset = Dataset.from_list(
        [tokenize_training_example(example, tokenizer, max_length=max_length) for example in train_examples]
    )
    eval_dataset = None
    if validation_examples:
        eval_dataset = Dataset.from_list(
            [tokenize_training_example(example, tokenizer, max_length=max_length) for example in validation_examples]
        )

    model = AutoModelForCausalLM.from_pretrained(resolved_model_path)
    if getattr(model.config, "vocab_size", 0) < len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_path),
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        logging_steps=1,
        save_strategy="no",
        eval_strategy="no" if eval_dataset is None else "epoch",
        report_to=[],
        remove_unused_columns=False,
        use_cpu=use_cpu,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=PromptCollator(pad_token_id=int(tokenizer.pad_token_id)),
    )

    result = trainer.train()
    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    summary = {
        "train_examples": len(train_examples),
        "validation_examples": len(validation_examples),
        "train_loss": float(result.training_loss),
        "output_dir": str(output_path),
    }
    (output_path / "training-summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SFT on prepared reset-aware JSONL artifacts.")
    parser.add_argument("--train", required=True, help="Path to the prepared training JSONL file.")
    parser.add_argument("--validation", help="Optional path to the prepared validation JSONL file.")
    parser.add_argument("--model", required=True, help="Model name or local model path.")
    parser.add_argument("--output-dir", required=True, help="Directory for checkpoints and summaries.")
    parser.add_argument("--max-length", type=int, default=1024, help="Maximum token length.")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--epochs", type=float, default=1.0, help="Number of epochs.")
    parser.add_argument("--train-batch-size", type=int, default=1, help="Per-device train batch size.")
    parser.add_argument("--eval-batch-size", type=int, default=1, help="Per-device eval batch size.")
    parser.add_argument(
        "--max-train-examples",
        type=int,
        default=None,
        metavar="N",
        help="Only train on the first N rows of the training JSONL (faster; not full-scale).",
    )
    parser.add_argument(
        "--strict-input-schema",
        dest="strict_input_schema",
        action="store_true",
        default=True,
        help="Reject any train/validation row that is not strict PromptExample JSONL.",
    )
    parser.add_argument(
        "--no-strict-input-schema",
        dest="strict_input_schema",
        action="store_false",
        help="Disable strict PromptExample schema validation for legacy corpora.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    summary = train_sft(
        train_path=args.train,
        validation_path=args.validation,
        model_name_or_path=args.model,
        output_dir=args.output_dir,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        max_train_examples=args.max_train_examples,
        strict_input_schema=args.strict_input_schema,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
