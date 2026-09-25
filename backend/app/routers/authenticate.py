"""POST /authenticate -- 1:1 verification against a claimed identity.

Identification (/identify) asks "who is this?" and ranks the whole gallery. Verification asks
"is this who they say they are?" and thresholds a single distance. Same frozen encoder, same
pooling -- only the downstream decision differs, per ARCHITECTURE.md's identification-vs-
verification split. This is the mode FAR/FRR/EER actually describe.

The threshold is GLOBAL and fixed in advance. It is not re-derived per person: that would be
an oracle a deployed system doesn't have. See eval/calibrate_auth.py for how it was measured
on held-out subjects.

GET /people backs the claim picker -- verification needs the user to name who they claim to be,
and background rows are not real people to claim.
"""

import numpy as np
from fastapi import APIRouter, HTTPException

from features.extract import extract_features

from ..embedding import embed
from ..gallery import get_all_entries, get_entry
from ..schemas import AuthenticateRequest, AuthenticateResponse, Person
from .identify import cosine_similarity
from .map import BACKGROUND_PREFIX

router = APIRouter()

# Calibrated on held-out subjects at the equal-error operating point, 5 enrollment sessions
# (what the UI collects). Produced by: python -m eval.calibrate_auth --checkpoint
# model/encoder_hard.pt -- see that script's docstring for why the per-subject EER reported in
# the README cannot be used here.
# Measured: EER 2.11% at this threshold over 1,000 held-out subjects (FRR 6.10% at FAR 1%).
AUTH_THRESHOLD = 0.49
OPERATING_POINT = "EER"


@router.get("/people", response_model=list[Person])
def people() -> list[Person]:
    return [
        Person(person_id=e["person_id"], name=e["name"])
        for e in get_all_entries()
        if not e["person_id"].startswith(BACKGROUND_PREFIX)
    ]


@router.post("/authenticate", response_model=AuthenticateResponse)
def authenticate(request: AuthenticateRequest) -> AuthenticateResponse:
    entry = get_entry(request.person_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="That person is not enrolled.")

    try:
        events = [event.model_dump() for event in request.events]
        query_embedding = embed(extract_features(events))
    except ValueError as exc:  # too few keystrokes (features.extract.MIN_KEYSTROKES)
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    similarity = cosine_similarity(query_embedding, np.array(entry["embedding"]))

    return AuthenticateResponse(
        person_id=entry["person_id"],
        name=entry["name"],
        accepted=similarity >= AUTH_THRESHOLD,
        similarity=similarity,
        threshold=AUTH_THRESHOLD,
        operating_point=OPERATING_POINT,
    )
