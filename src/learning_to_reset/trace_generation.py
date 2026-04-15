"""Generate verifiable trace records from Countdown samples and a source model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from learning_to_reset.countdown_verifier import score_countdown_response
from learning_to_reset.data import CountdownSample, load_countdown_samples
from learning_to_reset.model_resolver import resolve_model_name_or_path
from learning_to_reset.paper_sources import DEFAULT_TRACE_SOURCE_MODEL_ID, write_jsonl_records
from learning_to_reset.prompts import build_countdown_prompt


def _require_transformers():
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # pragma: no cover - runtime only
        raise RuntimeError(
            "transformers and torch are required to generate source traces."
        ) from exc
    return torch, AutoModelForCausalLM, AutoTokenizer


def _select_device(torch: Any, requested_device: str) -> str:
    if requested_device != "auto":
        return requested_device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_trace_record_from_response(
    sample: CountdownSample,
    response: str,
    *,
    source_model: str,
) -> Dict[str, Any]:
    """Turn one generated Countdown response into a trace record."""

    verification = score_countdown_response(response, sample)
    return {
        "source_id": sample.source_id,
        "problem": sample.question,
        "raw_trace": response,
        "is_correct": bool(verification.is_valid and verification.reaches_target),
        "metadata": {
            "source_model": source_model,
            "numbers": list(sample.numbers),
            "target": sample.target,
            "expression": verification.expression,
            "verification_reason": verification.reason,
            "is_valid": verification.is_valid,
            "reaches_target": verification.reaches_target,
        },
    }


def generate_trace_records(
    *,
    countdown_path: str | Path,
    output_path: str | Path,
    model_name_or_path: str = DEFAULT_TRACE_SOURCE_MODEL_ID,
    sample_limit: int | None = None,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
    device: str = "auto",
) -> Dict[str, Any]:
    """Generate raw trace records from Countdown questions using a source model."""

    torch, AutoModelForCausalLM, AutoTokenizer = _require_transformers()
    selected_device = _select_device(torch, device)
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
    model.to(selected_device)
    model.eval()

    samples = load_countdown_samples(countdown_path)
    if sample_limit is not None:
        samples = samples[:sample_limit]

    records: List[Dict[str, Any]] = []
    for sample in samples:
        prompt = build_countdown_prompt(sample, allow_clean=False)
        encoded = tokenizer(prompt, return_tensors="pt")
        encoded = {key: value.to(selected_device) for key, value in encoded.items()}

        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if temperature > 0:
            generation_kwargs["temperature"] = max(temperature, 1e-5)
            generation_kwargs["top_p"] = top_p

        with torch.no_grad():
            generated = model.generate(**encoded, **generation_kwargs)

        prompt_length = encoded["input_ids"].shape[1]
        token_ids = generated[0][prompt_length:]
        response = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
        records.append(
            build_trace_record_from_response(
                sample,
                response,
                source_model=model_name_or_path,
            )
        )

    write_jsonl_records(records, output_path)
    correct = sum(int(record["is_correct"]) for record in records)
    summary = {
        "model_name_or_path": model_name_or_path,
        "countdown_examples": len(samples),
        "correct_traces": correct,
        "incorrect_traces": len(records) - correct,
        "output_path": str(output_path),
        "device": selected_device,
    }
    (Path(output_path).with_suffix(".summary.json")).write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate verifiable Countdown trace records.")
    parser.add_argument("--countdown", required=True, help="Path to the flattened Countdown JSONL file.")
    parser.add_argument("--output-path", required=True, help="Path for the generated traces JSONL file.")
    parser.add_argument(
        "--model",
        default=DEFAULT_TRACE_SOURCE_MODEL_ID,
        help="Source model used to generate reasoning traces.",
    )
    parser.add_argument("--sample-limit", type=int, help="Optional sample limit.")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Maximum generation length.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling value.")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="Execution device.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    summary = generate_trace_records(
        countdown_path=args.countdown,
        output_path=args.output_path,
        model_name_or_path=args.model,
        sample_limit=args.sample_limit,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=args.device,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
