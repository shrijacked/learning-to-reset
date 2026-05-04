"""Reset-aware RLOO runtime for one-shot clean training and evaluation."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

from learning_to_reset.context_manager import response_requests_clean_retry
from learning_to_reset.countdown_verifier import score_countdown_response
from learning_to_reset.data import CountdownSample
from learning_to_reset.model_resolver import resolve_model_name_or_path
from learning_to_reset.prompts import PromptExample, parse_reasoning_prompt
from learning_to_reset.rollout_runtime import CleanTrajectory, build_clean_trajectory
from learning_to_reset.rloo import ModifiedRLOOResult, TrajectorySample, compute_modified_rloo_terms
from learning_to_reset.sft_runtime import load_prepared_examples


@dataclass(frozen=True)
class RolloutSegment:
    """A generated response segment together with its originating prompt."""

    prompt: str
    response: str
    token_ids: Tuple[int, ...]


@dataclass(frozen=True)
class RolloutCandidate:
    """One reset-aware interaction candidate for a prepared Countdown example."""

    sample: CountdownSample
    initial: RolloutSegment
    retry: RolloutSegment | None
    clean_trajectory: CleanTrajectory
    policy_trajectory: TrajectorySample

    @property
    def cleaned(self) -> bool:
        return self.clean_trajectory.cleaned

    @property
    def final_response(self) -> str:
        return self.clean_trajectory.final_response


def build_policy_trajectory_sample(
    clean_trajectory: CleanTrajectory,
    *,
    reward_mode: str = "correctness",
) -> TrajectorySample:
    """Project reward components onto the reward used by the policy update."""

    if reward_mode == "correctness":
        initial_reward = clean_trajectory.initial_reward.correctness_reward
        retry_reward = (
            clean_trajectory.retry_reward.correctness_reward
            if clean_trajectory.retry_reward is not None
            else None
        )
    elif reward_mode == "total":
        initial_reward = clean_trajectory.initial_reward.total_reward
        retry_reward = (
            clean_trajectory.retry_reward.total_reward
            if clean_trajectory.retry_reward is not None
            else None
        )
    else:
        raise ValueError("reward_mode must be 'correctness' or 'total'.")

    return TrajectorySample(
        initial_reward=initial_reward,
        initial_length=clean_trajectory.trajectory_sample.initial_length,
        retry_reward=retry_reward,
        retry_length=clean_trajectory.trajectory_sample.retry_length,
    )


def build_rollout_candidate(
    example: PromptExample,
    *,
    initial_response: str,
    initial_token_ids: Sequence[int],
    retry_response: str | None = None,
    retry_token_ids: Sequence[int] | None = None,
    reward_mode: str = "correctness",
) -> RolloutCandidate:
    """Attach reward and trajectory metadata to a sampled interaction."""

    prompt_parts = parse_reasoning_prompt(example.prompt)
    metadata = example.metadata
    sample = CountdownSample(
        numbers=tuple(int(number) for number in metadata["numbers"]),
        target=int(metadata["target"]),
        question=str(metadata.get("question", prompt_parts.question)),
        source_id=metadata.get("source_id"),
        solution=metadata.get("solution"),
    )

    retry_prompt = None
    if retry_response is not None or response_requests_clean_retry(initial_response):
        retry_prompt = "\n\n".join(
            part
            for part in (prompt_parts.base_instructions.strip(), f"Question: {prompt_parts.question}")
            if part
        )

    clean_trajectory = build_clean_trajectory(
        question=prompt_parts.question,
        sample=sample,
        initial_response=initial_response,
        retry_response=retry_response,
        base_instructions=prompt_parts.base_instructions,
        clean_instructions=prompt_parts.clean_instructions or "",
        initial_length=max(1, len(tuple(initial_token_ids))),
        retry_length=(
            max(1, len(tuple(retry_token_ids)))
            if retry_response is not None and retry_token_ids is not None
            else None
        ),
    )

    return RolloutCandidate(
        sample=sample,
        initial=RolloutSegment(
            prompt=example.prompt,
            response=initial_response,
            token_ids=tuple(int(token_id) for token_id in initial_token_ids),
        ),
        retry=(
            RolloutSegment(
                prompt=retry_prompt or "",
                response=retry_response,
                token_ids=tuple(int(token_id) for token_id in (retry_token_ids or ())),
            )
            if retry_response is not None
            else None
        ),
        clean_trajectory=clean_trajectory,
        policy_trajectory=build_policy_trajectory_sample(
            clean_trajectory,
            reward_mode=reward_mode,
        ),
    )


def summarize_rollout_candidates(
    rollouts: Sequence[RolloutCandidate],
    result: ModifiedRLOOResult,
) -> Dict[str, Any]:
    """Summarize one modified-RLOO batch for tracking and tests."""

    total = len(rollouts)
    clean_count = sum(int(rollout.cleaned) for rollout in rollouts)
    initial_tokens = sum(len(rollout.initial.token_ids) for rollout in rollouts)
    retry_tokens = sum(len(rollout.retry.token_ids) for rollout in rollouts if rollout.retry is not None)

    return {
        "responses_per_prompt": total,
        "clean_rate": (clean_count / total) if total else 0.0,
        "accuracy": (
            sum(int(rollout.clean_trajectory.final_reward.is_correct) for rollout in rollouts) / total
            if total
            else 0.0
        ),
        "valid_rate": (
            sum(int(rollout.clean_trajectory.final_reward.has_valid_format) for rollout in rollouts) / total
            if total
            else 0.0
        ),
        "average_score": (
            sum(rollout.clean_trajectory.final_reward.total_reward for rollout in rollouts) / total
            if total
            else 0.0
        ),
        "average_initial_tokens": (initial_tokens / total) if total else 0.0,
        "average_retry_tokens": (retry_tokens / clean_count) if clean_count else 0.0,
        "average_total_tokens": ((initial_tokens + retry_tokens) / total) if total else 0.0,
        "average_advantage": (
            sum(term.advantage for term in result.terms) / total if total else 0.0
        ),
        "normalization": result.normalization,
        "clean_trajectory_count": result.clean_trajectory_count,
    }


def compute_policy_loss(
    terms: Sequence[Any],
    initial_logprob_sums: Sequence[Any],
    retry_logprob_sums: Sequence[Any | None],
) -> Any:
    """Apply modified-RLOO scaling to segment log-probability sums."""

    if not (
        len(terms) == len(initial_logprob_sums) == len(retry_logprob_sums)
    ):
        raise ValueError("terms and log-probability sequences must have the same length.")

    total = 0
    for term, initial_logprob, retry_logprob in zip(
        terms,
        initial_logprob_sums,
        retry_logprob_sums,
    ):
        total = total - (term.initial_scale * initial_logprob)
        if retry_logprob is not None:
            total = total - (term.retry_scale * retry_logprob)
    return total


def _require_transformers():
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # pragma: no cover - runtime only
        raise RuntimeError(
            "transformers and torch are required for reset-aware RLOO training."
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


def inference_torch_dtype_for_device(torch: Any, device_str: str) -> Any:
    """Prefer bf16 on CUDA when supported; fp16 on CUDA/MPS otherwise; fp32 on CPU."""

    if device_str == "cuda":
        is_bf16 = getattr(torch.cuda, "is_bf16_supported", None)
        if callable(is_bf16) and is_bf16():
            return torch.bfloat16
        return torch.float16
    if device_str == "mps":
        return torch.float16
    return torch.float32


def _set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import torch
    except Exception:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _prepare_tokenizer(tokenizer: Any) -> Any:
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
    return tokenizer


def _trim_generated_ids(token_ids: Sequence[int], tokenizer: Any) -> Tuple[int, ...]:
    trimmed = []
    for token_id in token_ids:
        if tokenizer.eos_token_id is not None and int(token_id) == int(tokenizer.eos_token_id):
            break
        if tokenizer.pad_token_id is not None and int(token_id) == int(tokenizer.pad_token_id):
            break
        trimmed.append(int(token_id))
    return tuple(trimmed)


def build_generation_kwargs(
    tokenizer: Any,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    allow_clean: bool = True,
) -> Dict[str, Any]:
    """Build generation kwargs and optionally suppress another `<clean>` emission."""

    do_sample = temperature > 0
    generation_kwargs: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = max(temperature, 1e-5)
        generation_kwargs["top_p"] = top_p

    if not allow_clean:
        clean_token_ids = list(tokenizer.encode("<clean>", add_special_tokens=False))
        if clean_token_ids:
            generation_kwargs["bad_words_ids"] = [clean_token_ids]
    return generation_kwargs


def _generate_segment(
    *,
    model: Any,
    tokenizer: Any,
    prompt: str,
    device: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    allow_clean: bool = True,
) -> RolloutSegment:
    torch, _, _ = _require_transformers()
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}

    was_training = model.training
    model.eval()
    generation_kwargs = build_generation_kwargs(
        tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        allow_clean=allow_clean,
    )
    with torch.no_grad():
        generation = model.generate(
            **encoded,
            **generation_kwargs,
        )
    if was_training:
        model.train()

    prompt_length = encoded["input_ids"].shape[1]
    token_ids = _trim_generated_ids(generation[0][prompt_length:].tolist(), tokenizer)
    response = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
    return RolloutSegment(prompt=prompt, response=response, token_ids=token_ids)


def rollout_countdown_example(
    example: PromptExample,
    *,
    model: Any,
    tokenizer: Any,
    device: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    reward_mode: str = "correctness",
) -> RolloutCandidate:
    """Generate a one-shot clean-aware trajectory for one prompt."""

    initial = _generate_segment(
        model=model,
        tokenizer=tokenizer,
        prompt=example.prompt,
        device=device,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        allow_clean=True,
    )

    retry = None
    if response_requests_clean_retry(initial.response):
        prompt_parts = parse_reasoning_prompt(example.prompt)
        retry_prompt = "\n\n".join(
            part
            for part in (prompt_parts.base_instructions.strip(), f"Question: {prompt_parts.question}")
            if part
        )
        retry = _generate_segment(
            model=model,
            tokenizer=tokenizer,
            prompt=retry_prompt,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            allow_clean=False,
        )

    return build_rollout_candidate(
        example,
        initial_response=initial.response,
        initial_token_ids=initial.token_ids,
        retry_response=(retry.response if retry is not None else None),
        retry_token_ids=(retry.token_ids if retry is not None else None),
        reward_mode=reward_mode,
    )


def _response_logprob_sum(
    *,
    model: Any,
    tokenizer: Any,
    prompt: str,
    response_token_ids: Sequence[int],
    device: str,
) -> Any:
    torch, _, _ = _require_transformers()
    scored_token_ids = tuple(int(token_id) for token_id in response_token_ids)
    if not scored_token_ids:
        if tokenizer.eos_token_id is None:
            return torch.tensor(0.0, device=device)
        scored_token_ids = (int(tokenizer.eos_token_id),)

    encoded = tokenizer(prompt, return_tensors="pt")
    prompt_ids = encoded["input_ids"].to(device)
    response_ids = torch.tensor([list(scored_token_ids)], dtype=torch.long, device=device)
    input_ids = torch.cat([prompt_ids, response_ids], dim=1)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]
    target_ids = input_ids[:, 1:]
    token_logprobs = logits.log_softmax(dim=-1).gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)

    start = prompt_ids.shape[1] - 1
    stop = start + response_ids.shape[1]
    return token_logprobs[:, start:stop].sum()


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def evaluate_rollout_candidates(rollouts: Sequence[RolloutCandidate]) -> Dict[str, Any]:
    """Score final responses from reset-aware rollouts."""

    total = len(rollouts)
    correct = 0
    valid = 0
    total_score = 0.0
    cleaned = 0
    cleaned_score_total = 0.0
    results = []

    for rollout in rollouts:
        verification = score_countdown_response(rollout.final_response, rollout.sample)
        score = rollout.clean_trajectory.final_reward.total_reward
        correct += int(verification.is_valid and verification.reaches_target)
        valid += int(verification.is_valid)
        total_score += score
        cleaned += int(rollout.cleaned)
        if rollout.cleaned:
            cleaned_score_total += score
        results.append(
            {
                "source_id": rollout.sample.source_id,
                "prompt": rollout.initial.prompt,
                "initial_response": rollout.initial.response,
                "retry_response": rollout.retry.response if rollout.retry is not None else None,
                "final_response": rollout.final_response,
                "cleaned": rollout.cleaned,
                "expression": verification.expression,
                "value": str(verification.value) if verification.value is not None else None,
                "is_valid": verification.is_valid,
                "reaches_target": verification.reaches_target,
                "reason": verification.reason,
                "score": score,
            }
        )

    return {
        "total_examples": total,
        "correct": correct,
        "valid": valid,
        "accuracy": (correct / total) if total else 0.0,
        "valid_rate": (valid / total) if total else 0.0,
        "average_score": (total_score / total) if total else 0.0,
        "clean_rate": (cleaned / total) if total else 0.0,
        "score_when_cleaned": (cleaned_score_total / cleaned) if cleaned else 0.0,
        "results": results,
    }


def evaluate_reset_aware_model(
    examples: Sequence[PromptExample],
    *,
    model_name_or_path: str,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    top_p: float = 1.0,
    device: str = "auto",
) -> Dict[str, Any]:
    """Run reset-aware evaluation using the one-shot clean retry path."""

    torch, AutoModelForCausalLM, AutoTokenizer = _require_transformers()
    selected_device = _select_device(torch, device)
    resolved_model_path = resolve_model_name_or_path(model_name_or_path)
    tokenizer = _prepare_tokenizer(AutoTokenizer.from_pretrained(resolved_model_path))
    torch_dtype = inference_torch_dtype_for_device(torch, selected_device)
    model = AutoModelForCausalLM.from_pretrained(
        resolved_model_path,
        torch_dtype=torch_dtype,
    )
    if getattr(model.config, "vocab_size", 0) < len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
    model.to(selected_device)

    rollouts = [
        rollout_countdown_example(
            example,
            model=model,
            tokenizer=tokenizer,
            device=selected_device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        for example in examples
    ]
    summary = evaluate_rollout_candidates(rollouts)
    summary["device"] = selected_device
    return summary


def train_rloo(
    *,
    train_path: str | Path,
    output_dir: str | Path,
    model_name_or_path: str,
    validation_path: str | Path | None = None,
    steps: int = 10,
    prompts_per_step: int = 1,
    responses_per_prompt: int = 4,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    top_p: float = 1.0,
    learning_rate: float = 1e-6,
    weight_decay: float = 0.01,
    max_grad_norm: float = 1.0,
    reward_mode: str = "correctness",
    seed: int = 0,
    eval_every_steps: int = 0,
    device: str = "auto",
    max_train_examples: int | None = None,
    max_validation_examples: int | None = None,
) -> Dict[str, Any]:
    """Train a one-shot reset-aware policy with modified RLOO."""

    if responses_per_prompt < 2:
        raise ValueError("responses_per_prompt must be at least 2 for modified RLOO.")
    if prompts_per_step <= 0:
        raise ValueError("prompts_per_step must be positive.")
    if steps <= 0:
        raise ValueError("steps must be positive.")

    torch, AutoModelForCausalLM, AutoTokenizer = _require_transformers()
    _set_seed(seed)
    selected_device = _select_device(torch, device)

    resolved_model_path = resolve_model_name_or_path(model_name_or_path)
    tokenizer = _prepare_tokenizer(AutoTokenizer.from_pretrained(resolved_model_path))
    model = AutoModelForCausalLM.from_pretrained(resolved_model_path)
    if getattr(model.config, "vocab_size", 0) < len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
    model.to(selected_device)
    model.train()
    print(
        f"[rloo_runtime] model ready on {selected_device} ({resolved_model_path}); "
        "loading datasets next",
        flush=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    train_path = Path(train_path)
    cap_msg = (
        f", first {max_train_examples} rows only" if max_train_examples is not None else ""
    )
    print(
        f"[rloo_runtime] loading training examples from {train_path}{cap_msg}",
        flush=True,
    )
    if max_train_examples is not None and max_train_examples < 1:
        raise ValueError("max_train_examples must be >= 1 when provided")
    train_examples = load_prepared_examples(train_path, max_train_examples)
    if validation_path:
        val_path = Path(validation_path)
        vcap = (
            f", first {max_validation_examples} rows only"
            if max_validation_examples is not None
            else ""
        )
        print(f"[rloo_runtime] loading validation examples from {val_path}{vcap}", flush=True)
        if max_validation_examples is not None and max_validation_examples < 1:
            raise ValueError("max_validation_examples must be >= 1 when provided")
        validation_examples = load_prepared_examples(val_path, max_validation_examples)
    else:
        validation_examples = ()
    if not train_examples:
        raise ValueError("The training split is empty.")
    print(
        f"[rloo_runtime] loaded {len(train_examples)} train, "
        f"{len(validation_examples)} validation examples; starting {steps} step(s) on {selected_device}",
        flush=True,
    )

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    metrics_path = output_root / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    best_validation_accuracy = float("-inf")
    best_validation_summary: Dict[str, Any] | None = None
    cursor = 0
    last_loss = 0.0

    for step_index in range(1, steps + 1):
        print(
            f"[rloo_runtime] step {step_index}/{steps} (rollouts + backward; "
            f"first step can take several minutes)",
            flush=True,
        )
        batch_examples = []
        for _ in range(prompts_per_step):
            batch_examples.append(train_examples[cursor % len(train_examples)])
            cursor += 1

        optimizer.zero_grad()
        batch_loss = 0
        prompt_summaries = []

        for prompt_example in batch_examples:
            rollouts = [
                rollout_countdown_example(
                    prompt_example,
                    model=model,
                    tokenizer=tokenizer,
                    device=selected_device,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    reward_mode=reward_mode,
                )
                for _ in range(responses_per_prompt)
            ]
            result = compute_modified_rloo_terms(
                tuple(rollout.policy_trajectory for rollout in rollouts)
            )
            initial_logprob_sums = [
                _response_logprob_sum(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=rollout.initial.prompt,
                    response_token_ids=rollout.initial.token_ids,
                    device=selected_device,
                )
                for rollout in rollouts
            ]
            retry_logprob_sums = [
                (
                    _response_logprob_sum(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=rollout.retry.prompt,
                        response_token_ids=rollout.retry.token_ids,
                        device=selected_device,
                    )
                    if rollout.retry is not None
                    else None
                )
                for rollout in rollouts
            ]
            batch_loss = batch_loss + compute_policy_loss(
                result.terms,
                initial_logprob_sums,
                retry_logprob_sums,
            )
            prompt_summaries.append(summarize_rollout_candidates(rollouts, result))

        batch_loss = batch_loss / len(batch_examples)
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        last_loss = float(batch_loss.detach().cpu().item())
        step_metrics = {
            "step": step_index,
            "loss": last_loss,
            "accuracy": sum(item["accuracy"] for item in prompt_summaries) / len(prompt_summaries),
            "valid_rate": sum(item["valid_rate"] for item in prompt_summaries) / len(prompt_summaries),
            "average_score": sum(item["average_score"] for item in prompt_summaries) / len(prompt_summaries),
            "clean_rate": sum(item["clean_rate"] for item in prompt_summaries) / len(prompt_summaries),
            "average_initial_tokens": (
                sum(item["average_initial_tokens"] for item in prompt_summaries) / len(prompt_summaries)
            ),
            "average_retry_tokens": (
                sum(item["average_retry_tokens"] for item in prompt_summaries) / len(prompt_summaries)
            ),
            "average_total_tokens": (
                sum(item["average_total_tokens"] for item in prompt_summaries) / len(prompt_summaries)
            ),
            "average_advantage": (
                sum(item["average_advantage"] for item in prompt_summaries) / len(prompt_summaries)
            ),
            "reward_mode": reward_mode,
        }

        if validation_examples and eval_every_steps > 0 and step_index % eval_every_steps == 0:
            validation_summary = evaluate_rollout_candidates(
                [
                    rollout_countdown_example(
                        example,
                        model=model,
                        tokenizer=tokenizer,
                        device=selected_device,
                        max_new_tokens=max_new_tokens,
                        temperature=0.0,
                        top_p=1.0,
                    )
                    for example in validation_examples
                ]
            )
            step_metrics["validation_accuracy"] = validation_summary["accuracy"]
            step_metrics["validation_average_score"] = validation_summary["average_score"]
            step_metrics["validation_clean_rate"] = validation_summary["clean_rate"]

            if validation_summary["accuracy"] >= best_validation_accuracy:
                best_validation_accuracy = validation_summary["accuracy"]
                best_validation_summary = validation_summary
                best_dir = output_root / "best-checkpoint"
                model.save_pretrained(str(best_dir))
                tokenizer.save_pretrained(str(best_dir))
                (output_root / "best-validation-summary.json").write_text(
                    json.dumps(validation_summary, indent=2, ensure_ascii=True) + "\n",
                    encoding="utf-8",
                )

        _append_jsonl(metrics_path, step_metrics)

    final_dir = output_root / "final-checkpoint"
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    summary = {
        "train_examples": len(train_examples),
        "validation_examples": len(validation_examples),
        "steps": steps,
        "prompts_per_step": prompts_per_step,
        "responses_per_prompt": responses_per_prompt,
        "max_new_tokens": max_new_tokens,
        "reward_mode": reward_mode,
        "device": selected_device,
        "final_loss": last_loss,
        "best_validation_accuracy": (
            best_validation_summary["accuracy"] if best_validation_summary is not None else None
        ),
        "best_validation_average_score": (
            best_validation_summary["average_score"] if best_validation_summary is not None else None
        ),
        "output_dir": str(output_root),
    }
    (output_root / "training-summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a one-shot reset-aware RLOO policy.")
    parser.add_argument("--train", required=True, help="Prepared Countdown train JSONL file.")
    parser.add_argument("--validation", help="Optional prepared Countdown validation JSONL file.")
    parser.add_argument("--model", required=True, help="Model name or local model path.")
    parser.add_argument("--output-dir", required=True, help="Directory for checkpoints and metrics.")
    parser.add_argument("--steps", type=int, default=10, help="Number of optimization steps.")
    parser.add_argument(
        "--prompts-per-step",
        type=int,
        default=1,
        help="Number of prompts rolled out before each optimizer step.",
    )
    parser.add_argument(
        "--responses-per-prompt",
        type=int,
        default=4,
        help="Number of sampled trajectories per prompt.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Maximum generated tokens.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling value.")
    parser.add_argument("--learning-rate", type=float, default=1e-6, help="Optimizer learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Optimizer weight decay.")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Gradient clipping norm.")
    parser.add_argument(
        "--reward-mode",
        choices=("correctness", "total"),
        default="correctness",
        help="Reward component used for policy gradients.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--eval-every-steps",
        type=int,
        default=0,
        help="Run reset-aware validation every N steps.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="Execution device.",
    )
    parser.add_argument(
        "--max-train-examples",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Only load the first N rows of --train (faster startup; each step still "
            "uses prompts_per_step examples, cycling this pool)."
        ),
    )
    parser.add_argument(
        "--max-validation-examples",
        type=int,
        default=None,
        metavar="N",
        help="Only load the first N rows of --validation when present.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    summary = train_rloo(
        train_path=args.train,
        validation_path=args.validation,
        output_dir=args.output_dir,
        model_name_or_path=args.model,
        steps=args.steps,
        prompts_per_step=args.prompts_per_step,
        responses_per_prompt=args.responses_per_prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        reward_mode=args.reward_mode,
        seed=args.seed,
        eval_every_steps=args.eval_every_steps,
        device=args.device,
        max_train_examples=args.max_train_examples,
        max_validation_examples=args.max_validation_examples,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
