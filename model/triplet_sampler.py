"""Triplet sampler: draw (anchor, positive, negative) window triples from cached Aalto data.

Triplet construction (per ARCHITECTURE.md):
- Anchor (A): a random window from subject X
- Positive (P): a different window from the SAME subject X, but different session_id -- same person writing a different sentence
  (forces learning of person-specific rhythm, not sentence-specific timing)
- Negative (N): a window from a different subject Y (Y ≠ X)

Triples are sampled fresh per batch (not precomputed), so diversity is automatic.
"""

from __future__ import annotations

import numpy as np


class TripletSampler:
    """Load cached Aalto arrays and sample triplets for training."""

    def __init__(self, windows_path: str, mask_path: str, subject_ids_path: str, session_ids_path: str):
        """Load the four cached .npy arrays from Phase 1.

        Parameters
        ----------
        windows_path, mask_path, subject_ids_path, session_ids_path
            Paths to the .npy files from data/build_cache.py output.
        """
        # TODO: Load all four arrays with np.load()
        # self.windows = ...
        # self.mask = ...
        # self.subject_ids = ...
        # self.session_ids = ...
        

        self.windows = np.load(windows_path)
        self.mask = np.load(mask_path)
        self.subject_ids = np.load(subject_ids_path)
        self.session_ids = np.load(session_ids_path)

        self.subject_to_rows, self.subject_session_to_rows = self._build_subject_index()


    def _build_subject_index(self):
        """Build lookup: subject_id -> list of window row indexes (indeces)

        Also build: (subject_id, session_id) -> list of window row indexes (indeces)
        These enable fast triplet construction without scanning arrays repeatedly.
        """

    #--------------------------------------------------------------------------------------------------------
    #       index:       0          1          2          3          4          5          6      ...
    #       subject_id:  100001     100001     100001     100001     100001     100001     100001  ...
    #       session_id:  1090979    1091016    1091025    1091025    1091042    1091042    1091058 ...
    #       window:      (50,4)#0   (50,4)#1   (50,4)#2   (50,4)#3   (50,4)#4   (50,4)#5   (50,4)#6 ...
    #       mask (real): 48/50      32/50      50/50      9/50       50/50      3/50       50/50   ...
    #--------------------------------------------------------------------------------------------------------
    #
    #
    #   subject_to_rows["100001"] = [0, 1, 2, 3, 4, 5, 6, 7, ...]   # every row where subject_ids[i] == "100001"
    #
    #   subject_session_to_rows[("100001", "1091025")] = [2, 3]
    #   subject_session_to_rows[("100001", "1091042")] = [4, 5]
    #   subject_session_to_rows[("100001", "1091058")] = [6, 7]

        # TODO: Iterate over self.subject_ids and self.session_ids, building two dicts:
        #   subject_to_rows: {subject_id: [row0, row1, ...]}
        #   subject_session_to_rows: {(subject_id, session_id): [row0, row1, ...]}
        # This is a one-time overhead; called once in __init__.

        subject_to_rows = {}
        subject_session_to_rows = {}

        # Iterate over all rows in the dataset to populate the dictionaries
        for i in range(len(self.subject_ids)):

            subject_id = self.subject_ids[i]
            session_id = self.session_ids[i]

            # Update subject_to_rows

            if subject_id not in subject_to_rows:
                subject_to_rows[subject_id] = [] # create new entry in dict with empty list

            # else
            subject_to_rows[subject_id].append(i) # append to the list of existing entry 



            # Update subject_session_to_rows

            key = (subject_id, session_id)

            if key not in subject_session_to_rows:
                subject_session_to_rows[key] = [] # create new entry in dict with empty list

            # else
            subject_session_to_rows[key].append(i) # append to the list of existing entry


        return subject_to_rows, subject_session_to_rows


        

    def sample_triplet(self):
        """Draw one triplet: (anchor_window, anchor_mask, positive_window, ..., negative_window, negative_mask).

        Process:
        1. Pick a random subject X
        2. Pick a random session S1 for X (get anchor window)
        3. Pick a different session S2 for X (get positive window) -- must be S2 ≠ S1
        4. Pick a random different subject Y (Y ≠ X)
        5. Pick a random session for Y (get negative window)
        6. Return all three windows + masks as a dict or tuple

        Returns
        -------
        dict with keys: 'anchor_window', 'anchor_mask', 'positive_window', 'positive_mask',
                        'negative_window', 'negative_mask'
        Each value is a numpy array ready to convert to torch tensor.
        """
        # TODO: Implement the six-step process above.
        # Hints:
        #   - Use np.random.choice() to pick subjects/sessions
        #   - Use self.subject_to_rows and self.subject_session_to_rows to map to row indices
        #   - Index into self.windows and self.mask to get the actual data
        raise NotImplementedError

    def sample_batch(self, batch_size: int):
        """Sample a batch of triplets.

        Parameters
        ----------
        batch_size : int
            Number of triplets to sample

        Returns
        -------
        dict with keys: 'anchor_windows', 'anchor_masks', 'positive_windows', ..., 'negative_masks'
        Each value is a numpy array of shape (batch_size, 50, 4) or (batch_size, 50)
        ready to stack and convert to torch tensors for the forward pass.
        """
        # TODO: Call sample_triplet() batch_size times, collecting results.
        # Stack the individual windows/masks into batched arrays.
        raise NotImplementedError
