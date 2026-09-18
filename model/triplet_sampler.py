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

    def __init__(
        self,
        windows_path: str,
        mask_path: str,
        subject_ids_path: str,
        session_ids_path: str,
        subjects: set[str] | None = None,
    ):
        """Load the four cached .npy arrays from Phase 1.

        Parameters
        ----------
        windows_path, mask_path, subject_ids_path, session_ids_path
            Paths to the .npy files from data/build_cache.py output.
        subjects
            If given, only these subjects are indexed (and so ever sampled).
            Used for the subject-disjoint train/eval split; the arrays
            themselves are not filtered, so row indices still point into the
            full cache.
        """
        self.allowed_subjects = subjects
        # TODO: Load all four arrays with np.load()
        # self.windows = ...
        # self.mask = ...
        # self.subject_ids = ...
        # self.session_ids = ...
        

        self.windows = np.load(windows_path)
        self.mask = np.load(mask_path)
        self.subject_ids = np.load(subject_ids_path)
        self.session_ids = np.load(session_ids_path)

        self.subject_to_rows, self.subject_session_to_rows, self.subject_to_sessions = self._build_subject_index()

        # Only subjects with 2+ distinct sessions can serve as anchors (need a
        # different session for the positive).
        self.eligible_anchor_subjects = [
            subj for subj, sessions in self.subject_to_sessions.items() if len(sessions) >= 2
        ]

        # Precomputed once so sample_triplet doesn't rebuild/rescan this list
        # on every call.
        self.all_subject_ids = np.array(list(self.subject_to_rows.keys()))


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
            if self.allowed_subjects is not None and subject_id not in self.allowed_subjects:
                continue
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


        # Build subject_id -> list of distinct session_ids from the keys we just
        # collected (cheap: one pass over unique (subject, session) pairs, not
        # the full array).
        subject_to_sessions = {}
        for subj, session_id in subject_session_to_rows.keys():
            subject_to_sessions.setdefault(subj, []).append(session_id)

        return subject_to_rows, subject_session_to_rows, subject_to_sessions


        

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
        
    
        # Step 1: Pick a random subject X (must have 2+ sessions, so a distinct
        # positive session exists)
        anchor = np.random.choice(self.eligible_anchor_subjects)

        # Step 2: Pick a random session S1 for X (get anchor window)
        sessions_for_anchor = self.subject_to_sessions[anchor]
        session1 = np.random.choice(sessions_for_anchor)

        # Step 3: Pick a different session S2 for X (get positive window) -- must be S2 ≠ S1
        remaining_sessions = [s for s in sessions_for_anchor if s != session1]
        session2 = np.random.choice(remaining_sessions)

        # Step 4: Pick a random different subject Y (Y ≠ X)
        negative_subject = np.random.choice(self.all_subject_ids)
        while negative_subject == anchor:
            negative_subject = np.random.choice(self.all_subject_ids)

        # Step 5: Pick a random session for Y (get negative window)
        sessions_for_negative = self.subject_to_sessions[negative_subject]
        negative_session = np.random.choice(sessions_for_negative)

        # Step 6: Return all three windows + masks as a dict or tuple
        anchor_rows = self.subject_session_to_rows[(anchor, session1)]
        positive_rows = self.subject_session_to_rows[(anchor, session2)]
        negative_rows = self.subject_session_to_rows[(negative_subject, negative_session)]

        anchor_row = np.random.choice(anchor_rows)
        positive_row = np.random.choice(positive_rows)
        negative_row = np.random.choice(negative_rows)

        return {
            'anchor_window': self.windows[anchor_row],
            'anchor_mask': self.mask[anchor_row],
            'positive_window': self.windows[positive_row],
            'positive_mask': self.mask[positive_row],
            'negative_window': self.windows[negative_row],
            'negative_mask': self.mask[negative_row]
        }

        








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
        triplets = [self.sample_triplet() for _ in range(batch_size)]

        return {
            'anchor_windows': np.stack([t['anchor_window'] for t in triplets]),
            'anchor_masks': np.stack([t['anchor_mask'] for t in triplets]),
            'positive_windows': np.stack([t['positive_window'] for t in triplets]),
            'positive_masks': np.stack([t['positive_mask'] for t in triplets]),
            'negative_windows': np.stack([t['negative_window'] for t in triplets]),
            'negative_masks': np.stack([t['negative_mask'] for t in triplets]),
        }
