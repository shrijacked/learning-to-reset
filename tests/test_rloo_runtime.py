import math
import unittest

from learning_to_reset.context_manager import build_retry_prompt
from learning_to_reset.prompts import DEFAULT_BASE_INSTRUCTIONS, build_reasoning_prompt
from learning_to_reset.rloo import compute_modified_rloo_terms
from learning_to_reset.rloo_runtime import (
    build_generation_kwargs,
    build_rollout_candidate,
    compute_policy_loss,
    summarize_rollout_candidates,
)


class RLOORuntimeTests(unittest.TestCase):
    def test_build_generation_kwargs_blocks_clean_token_when_not_allowed(self) -> None:
        class TokenizerStub:
            pad_token_id = 0
            eos_token_id = 1

            def encode(self, text, add_special_tokens=False):
                if text == "<clean>" and not add_special_tokens:
                    return [42]
                return [7]

        kwargs = build_generation_kwargs(
            TokenizerStub(),
            max_new_tokens=64,
            temperature=0.0,
            top_p=1.0,
            allow_clean=False,
        )

        self.assertEqual(kwargs["bad_words_ids"], [[42]])

    def test_build_rollout_candidate_uses_correctness_reward_for_policy_by_default(self) -> None:
        example = {
            "prompt": build_reasoning_prompt(
                "Reach 68 using 60, 27, 19.",
                allow_clean=True,
            ),
            "response": "",
            "metadata": {
                "source_id": "c1",
                "numbers": (60, 27, 19),
                "target": 68,
                "question": "Reach 68 using 60, 27, 19.",
            },
        }

        candidate = build_rollout_candidate(
            type("PromptExampleStub", (), example)(),
            initial_response="<think>Confusing.</think><clean>",
            initial_token_ids=(1, 2, 3),
            retry_response="<think>Fresh.</think><answer>(60 + 27) - 19</answer>",
            retry_token_ids=(4, 5, 6, 7),
        )

        self.assertTrue(candidate.cleaned)
        self.assertEqual(
            candidate.retry.prompt,
            build_retry_prompt("Reach 68 using 60, 27, 19.", DEFAULT_BASE_INSTRUCTIONS),
        )
        self.assertTrue(math.isclose(candidate.clean_trajectory.final_reward.total_reward, 1.1))
        self.assertTrue(math.isclose(candidate.policy_trajectory.retry_reward or 0.0, 1.0))

    def test_build_rollout_candidate_can_use_total_reward_for_policy(self) -> None:
        example = {
            "prompt": build_reasoning_prompt(
                "Reach 68 using 60, 27, 19.",
                allow_clean=True,
            ),
            "response": "",
            "metadata": {
                "source_id": "c2",
                "numbers": (60, 27, 19),
                "target": 68,
                "question": "Reach 68 using 60, 27, 19.",
            },
        }

        candidate = build_rollout_candidate(
            type("PromptExampleStub", (), example)(),
            initial_response="<think>Confusing.</think><clean>",
            initial_token_ids=(1, 2),
            retry_response="<think>Fresh.</think><answer>(60 + 27) - 19</answer>",
            retry_token_ids=(3, 4, 5),
            reward_mode="total",
        )

        self.assertTrue(math.isclose(candidate.policy_trajectory.retry_reward or 0.0, 1.1))

    def test_compute_policy_loss_matches_segment_scales(self) -> None:
        class Term:
            def __init__(self, initial_scale: float, retry_scale: float) -> None:
                self.initial_scale = initial_scale
                self.retry_scale = retry_scale

        loss = compute_policy_loss(
            [Term(0.5, 0.25), Term(-0.2, 0.0)],
            [2.0, 3.0],
            [1.0, None],
        )

        self.assertTrue(math.isclose(loss, -0.65))

    def test_summarize_rollout_candidates_reports_clean_and_score(self) -> None:
        base_prompt = build_reasoning_prompt("Reach 68 using 60, 27, 19.", allow_clean=True)

        cleaned = build_rollout_candidate(
            type(
                "PromptExampleStub",
                (),
                {
                    "prompt": base_prompt,
                    "response": "",
                    "metadata": {
                        "source_id": "c3",
                        "numbers": (60, 27, 19),
                        "target": 68,
                        "question": "Reach 68 using 60, 27, 19.",
                    },
                },
            )(),
            initial_response="<think>Confusing.</think><clean>",
            initial_token_ids=(1, 2, 3),
            retry_response="<think>Fresh.</think><answer>(60 + 27) - 19</answer>",
            retry_token_ids=(4, 5, 6),
        )
        direct = build_rollout_candidate(
            type(
                "PromptExampleStub",
                (),
                {
                    "prompt": build_reasoning_prompt("Reach 55 using 61, 63, 57.", allow_clean=True),
                    "response": "",
                    "metadata": {
                        "source_id": "c4",
                        "numbers": (61, 63, 57),
                        "target": 55,
                        "question": "Reach 55 using 61, 63, 57.",
                    },
                },
            )(),
            initial_response="<think>Still wrong.</think>",
            initial_token_ids=(7, 8),
        )

        result = compute_modified_rloo_terms((cleaned.policy_trajectory, direct.policy_trajectory))
        summary = summarize_rollout_candidates((cleaned, direct), result)

        self.assertAlmostEqual(summary["clean_rate"], 0.5)
        self.assertAlmostEqual(summary["accuracy"], 0.5)
        self.assertAlmostEqual(summary["average_score"], 0.55)
        self.assertEqual(summary["clean_trajectory_count"], 1)


if __name__ == "__main__":
    unittest.main()
