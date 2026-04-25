# Short Project Update

As of 24 April 2026, the repository now supports the core workflow end to end: paper-aligned source preparation, reset-aware supervised fine-tuning, reset-aware evaluation, and reset-aware RLOO. We have also implemented the planned extension path for bounded multi-clean control, selective retention across resets, and recall-aware memory.

The current local audit confirms that the engineering pipeline is working and tested. The full unit test suite currently passes `123/123` tests. We now also have hard-problem evaluation, verifier-based scoring, and failure-mined recovery traces for the next training cycle.

The main experimental signal so far is that the reset mechanism improves answer validity much more than it improves arithmetic correctness. On the freshest `32`-example hard Countdown holdout, raw one-pass decoding was `0/32` valid and `0/32` correct, while reset-aware retry reached `29/32` valid and `1/32` correct.

The immediate next steps are to strengthen arithmetic-grounded recovery supervision, train the next SFT checkpoint on those improved traces, rerun reset-aware RLOO once the correctness signal is stronger, and then scale the hard-slice and extension comparisons.
