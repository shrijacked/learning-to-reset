# Current Approaches

## Main Families of Existing Solutions

### 1. Memory-Augmented Systems

- Maintain notes, graphs, or retrieved facts outside the active context window.
- Help with long-horizon tasks, but much of the context-management logic lives in an external controller.
- Useful reference point for later extensions involving write and recall actions.

### 2. Context Compression

- Replace or summarize long reasoning histories so more information fits into the context window.
- Reduces context pressure, but does not directly decide when an explored path should be abandoned.
- Best viewed as complementary to reset-aware reasoning rather than a full substitute.

### 3. Reasoning Training With Search Behaviors

- Supervised traces and RL objectives can teach verification, backtracking, and better search habits.
- Strong baseline for mathematical reasoning tasks such as Countdown.
- Still leaves open the question of whether the model can learn an explicit cleanup action for stale context.

## Gap This Project Targets

The project focuses on self-managed context cleanup:

- let the model detect when the current search is becoming confusing
- give it a direct reset action through `<clean>`
- reward the full interaction so the reset decision and the post-reset answer are learned together

## Why This Matters

- Hard arithmetic search tasks often fail because the model keeps building on bad intermediate steps.
- An explicit reset action is simpler to interpret than full memory systems and easier to extend in stages.
- Once the one-shot baseline works, the same framework can grow toward multi-step cleaning, selective retention, and recall-aware memory.
