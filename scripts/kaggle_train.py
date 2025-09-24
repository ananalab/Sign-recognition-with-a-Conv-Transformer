"""Run a list of training experiments on a Kaggle GPU and fetch the results.

The current source tree (src/, configs/, splits/) is embedded in the kernel script as a tarball,
so what runs on Kaggle is exactly the local code. Data come from Kaggle sources:
the conversion kernel output (asl-signs) and/or a private dataset with the processed LSFB data.

    uv run python scripts/kaggle_train.py --experiments experiments/asl_main.yaml --push
    uv run python scripts/kaggle_train.py --experiments experiments/asl_main.yaml --fetch

Experiment file (YAML)::

    name: asl-main               # kernel slug suffix
    sources: [kaggle_asl]        # kaggle_asl and/or lsfb_isol
    pretrained_from: []          # optional kernels whose models/*/best.pt are made available
    runs:
      - run_name: asl-transformer-s0
        overrides: [seed=0]
        evaluate: [val, test]    # optional
"""

