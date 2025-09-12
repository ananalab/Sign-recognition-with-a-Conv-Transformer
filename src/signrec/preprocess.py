"""Preprocessing: raw unified sequence ``[T, L, 3]`` to model features ``[T_out, F]`` + mask.

:func:`preprocess` is used for training, evaluation and, through a line-by-line JavaScript port
checked by fixtures, the browser demo. All steps are pure numpy.

Order of operations: trim, canonical hand (mirror), spatial normalisation, fixed length,
features, NaN to 0.
"""

from __future__ import annotations

