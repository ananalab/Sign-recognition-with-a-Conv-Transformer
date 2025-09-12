"""Read Google ISLR (Kaggle ``asl-signs``) parquet files as unified ``[T, L, 3]`` arrays.

Each parquet has one row per (frame, landmark) with columns
``frame, row_id, type, landmark_index, x, y, z``; ``type`` in {face, left_hand, pose, right_hand}.
Frames hold 543 landmarks in the order face (468) | left_hand (21) | pose (33) | right_hand (21).
"""

