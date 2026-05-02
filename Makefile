PYTHON ?= python3
PYTHONPATH ?= src
BASE_MODEL ?= Qwen/Qwen2.5-0.5B
OUT_DIR ?= tmp/replicate-paper-dryrun

.PHONY: demo figures test replicate-pilot replicate-paper

demo:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m learning_to_reset

figures:
	PYTHONPATH=$(PYTHONPATH):. $(PYTHON) scripts/build_figures.py --output-dir docs/figures

test:
	PYTHONPATH=$(PYTHONPATH):. $(PYTHON) -m unittest discover -s tests

replicate-pilot:
	LTR_REPLICATE_DRY_RUN=1 PYTHONPATH=$(PYTHONPATH) bash scripts/replicate_paper.sh --dry-run --base-model $(BASE_MODEL) --out-dir $(OUT_DIR)

replicate-paper:
	PYTHONPATH=$(PYTHONPATH) bash scripts/replicate_paper.sh --base-model $(BASE_MODEL) --scale-override paper --out-dir $(OUT_DIR)
