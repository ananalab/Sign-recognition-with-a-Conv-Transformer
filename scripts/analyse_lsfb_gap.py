"""Why is LSFB validation accuracy higher than test accuracy? -> reports/lsfb_gap.json.

Validation signers are drawn from the official training signers, test signers are the official
test signers. The script compares the two sets on factors that can explain an accuracy gap:

- recording sessions: LSFB clips come from dialogues between two signers; a held-out signer whose
  dialogue partner is a training signer shares the conversation (topics, signs) with the
  training data;
- data: clips per signer, durations, missing hands, class distribution;
- predictions (three seeds of the from-scratch model, all data): accuracy of each split, per
  clip and per signer, test accuracy re-weighted to the validation class distribution, and
  accuracy of signers whose partner is or is not in the training set.

    uv run python scripts/analyse_lsfb_gap.py [--run lsfb-scratch-nall]
"""

import numpy as np
import pandas as pd

