"""Countdown generation/evaluation runtime built on prepared artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from learning_to_reset.countdown_verifier import score_countdown_response
from learning_to_reset.data import CountdownSample
from learning_to_reset.model_resolver import resolve_model_name_or_path
from learning_to_reset.prompts import PromptExample
from learning_to_reset.sft_runtime import load_prepared_examples


def build_countdown_sample_from_example(example: PromptExample) -> CountdownSample:
    """Reconstruct a Countdown sample from prepared prompt metadata."""

    metadata = example.metadata
    return CountdownSample(
        numbers=tuple(int(number) for number in metadata["numbers"]),
        target=int(metadata["target"]),
        question=str(metadata.get("question", example.prompt)),
        source_id=metadata.get("source_id"),
        solution=metadata.get("solution"),
    )


def evaluate_countdown_outputs(
    examples: Sequence[PromptExample],
    responses: Sequence[str],
) -> Dict[str, Any]:
    """Score generated responses against Countdown prompts and summarize them."""

    if len(examples) != len(responses):
        raise ValueError("examples and responses must have the same length.")

    per_example = []
    correct = 0
    valid = 0
    total_score = 0.0
    for example, response in zip(examples, responses):
        sample = build_countdown_sample_from_example(example)
        verification = score_countdown_response(response, sample)
        score = (0.1 if verification.is_valid else 0.0) + (
            1.0 if verification.is_valid and verification.reaches_target else 0.0
        )
        valid += int(verification.is_valid)
        correct += int(verification.is_valid and verification.reaches_target)
        total_score += score
        per_example.append(
            {
                "source_id": sample.source_id,
                "prompt": example.prompt,
                "response": response,
                "expression": verification.expression,
                "value": str(verification.value) if verification.value is not None else None,
                "is_valid": verification.is_valid,
                "reaches_target": verification.reaches_target,
                "reason": verification.reason,
                "score": score,
            }
        )

    total = len(examples)
    return {
        "total_examples": total,
        "correct": correct,
        "valid": valid,
        "accuracy": (correct / total) if total else 0.0,
        "valid_rate": (valid / total) if total else 0.0,
        "average_score": (total_score / total) if total else 0.0,
        "results": per_example,
    }


def _require_transformers():
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # pragma: no cover - runtime only
        raise RuntimeError(
            "transformers/torch are required for generation evaluation. "
            "Install them in the active environment first."
        ) from exc
    return torch, AutoModelForCausalLM, AutoTokenizer


def generate_countdown_responses(
    examples: Sequence[PromptExample],
    *,
    model_name_or_path: str,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
) -> List[str]:
    """Generate responses for prepared Countdown prompts."""

    torch, AutoModelForCausalLM, AutoTokenizer = _require_transformers()
    resolved_model_path = resolve_model_name_or_path(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(resolved_model_path)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})

    model = AutoModelForCausalLM.from_pretrained(resolved_model_path)
    if getattr(model.config, "vocab_size", 0) < len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))

    responses = []
    for example in examples:
        encoded = tokenizer(example.prompt, return_tensors="pt")
        with torch.no_grad():
            generation = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        prompt_length = encoded["input_ids"].shape[1]
        generated_ids = generation[0][prompt_length:]
        responses.append(tokenizer.decode(generated_ids, skip_special_tokens=True).strip())
    return responses


def write_evaluation_outputs(summary: Dict[str, Any], output_dir: str | Path) -> None:
    """Write evaluation summary and per-example results to disk."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    summary_payload = {key: value for key, value in summary.items() if key != "results"}
    (destination / "summary.json").write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    lines = [json.dumps(result, ensure_ascii=True) for result in summary["results"]]
    payload = "\n".join(lines)
    if payload:
        payload += "\n"
    (destination / "results.jsonl").write_text(payload, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate and score Countdown responses.")
    parser.add_argument("--prepared-countdown", required=True, help="Prepared Countdown JSONL file.")
    parser.add_argument("--model", required=True, help="Model name or local model path.")
    parser.add_argument("--output-dir", required=True, help="Directory for evaluation outputs.")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Generation length.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument(
        "--raw-generation",
        action="store_true",
        help="Skip the clean-aware retry path and score only the first generation.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    examples = load_prepared_examples(args.prepared_countdown)
    if args.raw_generation:
        responses = generate_countdown_responses(
            examples,
            model_name_or_path=args.model,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        summary = evaluate_countdown_outputs(examples, responses)
    else:
        from learning_to_reset.rloo_runtime import evaluate_reset_aware_model

        summary = evaluate_reset_aware_model(
            examples,
            model_name_or_path=args.model,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
    write_evaluation_outputs(summary, args.output_dir)
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key != "results"},
            indent=2,
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
