"""Countdown generation/evaluation runtime built on prepared artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from learning_to_reset.context_manager import response_requests_clean_retry
from learning_to_reset.countdown_verifier import VerificationResult, score_countdown_response
from learning_to_reset.data import CountdownSample
from learning_to_reset.model_resolver import resolve_model_name_or_path
from learning_to_reset.prompts import PromptExample, parse_reasoning_prompt
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
    cleaned_count = 0
    cleaned_score_total = 0.0
    for example, response in zip(examples, responses):
        sample = build_countdown_sample_from_example(example)
        verification = score_countdown_response(response, sample)
        score = (0.1 if verification.is_valid else 0.0) + (
            1.0 if verification.is_valid and verification.reaches_target else 0.0
        )
        cleaned = "<clean>" in response.lower()
        valid += int(verification.is_valid)
        correct += int(verification.is_valid and verification.reaches_target)
        total_score += score
        if cleaned:
            cleaned_count += 1
            cleaned_score_total += score
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
                "cleaned": cleaned,
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
        "clean_rate": (cleaned_count / total) if total else 0.0,
        "score_when_cleaned": (cleaned_score_total / cleaned_count) if cleaned_count else 0.0,
        "results": per_example,
    }


def _build_retry_prompt(example: PromptExample) -> str:
    parts = parse_reasoning_prompt(example.prompt)
    pieces = [parts.base_instructions.strip(), f"Question: {parts.question}"]
    return "\n\n".join(piece for piece in pieces if piece)


def format_verifier_feedback(verification: VerificationResult, sample: CountdownSample) -> str:
    """Build a short block appended to retry prompts with external arithmetic truth.

    This is a decode-time hint: it does not change how ``<answer>`` is scored, but
    gives the model an explicit value-vs-target signal after a failed segment.
    """

    nums = ", ".join(str(n) for n in sample.numbers)
    if not verification.is_valid:
        return (
            "Verifier feedback: the last response could not be scored as a legal Countdown "
            f"answer ({verification.reason}) Target is {sample.target} using numbers [{nums}]. "
            "Reply with a single arithmetic expression inside <answer> that uses each given "
            "number at most once, then stop or use <clean> only if you must reset again."
        )
    if verification.value is not None:
        return (
            "Verifier feedback: your last <answer> expression evaluates to "
            f"{verification.value}, but the target is {sample.target}. "
            f"Numbers (each at most once): [{nums}]. Revise the expression inside <answer>."
        )
    return (
        f"Verifier feedback: target is {sample.target}, numbers [{nums}]. "
        "Provide a corrected expression inside <answer>."
    )


def run_multi_clean_eval_loop(
    examples: Sequence[PromptExample],
    *,
    generator: Callable[[str, bool], str],
    max_clean_tries: int,
    clean_token: str = "<clean>",
    verifier_feedback: bool = False,
) -> Dict[str, Any]:
    """Evaluate examples with verifier-aware multi-clean retry decoding.

    The first segment is generated against the original prompt with
    ``allow_clean=True``. After each segment, the response is verified; the
    first segment to reach the target wins, regardless of any subsequent
    cleans the model might still emit. If the segment requests a clean and
    budget remains, a retry prompt is built and the next segment is
    generated. The very last segment under the budget is generated with
    ``allow_clean=False`` to force the model to commit to an answer.

    When ``verifier_feedback`` is true, each retry prompt appends a compact
    verifier summary (computed value vs target, or invalid reason) so the
    model is not blindly re-prompted with the same question alone.
    """

    if max_clean_tries < 1:
        raise ValueError("max_clean_tries must be >= 1.")

    per_example: List[Dict[str, Any]] = []
    correct = 0
    valid = 0
    total_score = 0.0
    cleaned_count = 0
    cleaned_score_total = 0.0

    for example in examples:
        sample = build_countdown_sample_from_example(example)
        retry_prompt = _build_retry_prompt(example)
        segments: List[Dict[str, Any]] = []
        clean_count = 0
        winning_index: Optional[int] = None

        prompt = example.prompt
        allow_clean = True
        for _attempt in range(max_clean_tries + 1):
            response = generator(prompt, allow_clean)
            verification = score_countdown_response(response, sample)
            requested_clean = response_requests_clean_retry(
                response, clean_token=clean_token
            )
            segments.append(
                {
                    "prompt": prompt,
                    "response": response,
                    "is_valid": verification.is_valid,
                    "reaches_target": verification.reaches_target,
                    "requested_clean": requested_clean,
                }
            )

            if verification.is_valid and verification.reaches_target:
                winning_index = len(segments) - 1
                break

            if not requested_clean:
                break

            if clean_count >= max_clean_tries:
                break

            clean_count += 1
            if verifier_feedback:
                prompt = f"{retry_prompt}\n\n{format_verifier_feedback(verification, sample)}"
            else:
                prompt = retry_prompt
            allow_clean = clean_count < max_clean_tries

        budget_exhausted = winning_index is None and clean_count >= max_clean_tries

        final = segments[winning_index] if winning_index is not None else segments[-1]
        final_response = final["response"]
        verification = score_countdown_response(final_response, sample)
        score = (0.1 if verification.is_valid else 0.0) + (
            1.0 if verification.is_valid and verification.reaches_target else 0.0
        )
        any_cleaned = any(segment["requested_clean"] for segment in segments)
        valid += int(verification.is_valid)
        correct += int(verification.is_valid and verification.reaches_target)
        total_score += score
        if any_cleaned:
            cleaned_count += 1
            cleaned_score_total += score

        per_example.append(
            {
                "source_id": sample.source_id,
                "prompt": example.prompt,
                "response": final_response,
                "expression": verification.expression,
                "value": str(verification.value) if verification.value is not None else None,
                "is_valid": verification.is_valid,
                "reaches_target": verification.reaches_target,
                "reason": verification.reason,
                "score": score,
                "cleaned": any_cleaned,
                "clean_count": clean_count,
                "segment_count": len(segments),
                "budget_exhausted": budget_exhausted,
                "segments": segments,
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
        "clean_rate": (cleaned_count / total) if total else 0.0,
        "score_when_cleaned": (cleaned_score_total / cleaned_count) if cleaned_count else 0.0,
        "max_clean_tries": max_clean_tries,
        "verifier_feedback": verifier_feedback,
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


def evaluate_with_multi_clean_decoder(
    examples: Sequence[PromptExample],
    *,
    model_name_or_path: str,
    max_new_tokens: int = 384,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_clean_tries: int = 1,
    device: str = "auto",
    verifier_feedback: bool = False,
) -> Dict[str, Any]:
    """Evaluate with verifier-aware multi-clean retry decoding."""

    from learning_to_reset.rloo_runtime import (
        _generate_segment,
        _prepare_tokenizer,
        _require_transformers as _rloo_require,
        _select_device,
    )

    torch, AutoModelForCausalLM, AutoTokenizer = _rloo_require()
    selected_device = _select_device(torch, device)
    resolved_model_path = resolve_model_name_or_path(model_name_or_path)
    tokenizer = _prepare_tokenizer(AutoTokenizer.from_pretrained(resolved_model_path))
    model = AutoModelForCausalLM.from_pretrained(resolved_model_path)
    if getattr(model.config, "vocab_size", 0) < len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
    model.to(selected_device)

    def generator(prompt: str, allow_clean: bool) -> str:
        segment = _generate_segment(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=selected_device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            allow_clean=allow_clean,
        )
        return segment.response

    summary = run_multi_clean_eval_loop(
        examples,
        generator=generator,
        max_clean_tries=max_clean_tries,
        verifier_feedback=verifier_feedback,
    )
    summary["device"] = selected_device
    summary["model"] = resolved_model_path
    return summary


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
    parser.add_argument(
        "--max-clean-tries",
        type=int,
        default=1,
        help=(
            "Maximum number of <clean> emissions allowed per example before the "
            "decoder must commit. 1 (default) preserves the legacy one-shot reset "
            "path; values >= 2 enable the verifier-aware multi-clean decoder."
        ),
    )
    parser.add_argument(
        "--verifier-feedback",
        action="store_true",
        help=(
            "With --max-clean-tries >= 2, append verifier truth (evaluated value vs "
            "target, or invalid reason) to each retry prompt after <clean>. "
            "Improves retry signal; not part of the original paper baseline."
        ),
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
    elif args.max_clean_tries >= 2:
        summary = evaluate_with_multi_clean_decoder(
            examples,
            model_name_or_path=args.model,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            max_clean_tries=args.max_clean_tries,
            verifier_feedback=args.verifier_feedback,
        )
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
