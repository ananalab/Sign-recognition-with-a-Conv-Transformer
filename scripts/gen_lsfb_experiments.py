"""Write experiments/lsfb_{scratch,transfer}.yaml: LSFB learning curves (scratch vs. ASL).

The number of epochs grows as the training set shrinks so that every run performs a
comparable number of updates; small runs load data in the main process.

    uv run python scripts/gen_lsfb_experiments.py
"""

