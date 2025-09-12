import numpy as np
import pandas as pd

from signrec.metrics import confused_pairs, confusion, per_group_accuracy, summary, topk_accuracy
from signrec.splits import random_split, signer_split, split_indices


def _index(n_signers: int = 10, per: int = 20) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(n_signers * per)],
            "signer_id": np.repeat([f"P{k}" for k in range(n_signers)], per),
        }
    )


