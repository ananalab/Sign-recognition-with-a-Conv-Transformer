"""Stream LSFB-ISOL poses from the UNamur server into the packed unified format.

Only the unified landmark subset of each instance is kept (a few KB per clip), so the
~10 GB of raw poses never touch the disk. Interrupted runs resume from ``_cache/``.

    uv run python scripts/convert_lsfb.py --config configs/data/lsfb_isol.yaml [--limit 500]
"""

from __future__ import annotations

