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

from ..embedding import embed_sessions
from ..gallery import get_all_entries, get_entry
from ..schemas import PROMPT_COUNTS, AuthenticateRequest, AuthenticateResponse, Person
from ..thresholds import OPERATING_POINT, auth_threshold
from .identify import cosine_similarity
from .map import BACKGROUND_PREFIX

router = APIRouter()


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

    query_prompts = len(request.sessions)
    if query_prompts not in PROMPT_COUNTS:
        raise HTTPException(status_code=422, detail=f"Verify with {PROMPT_COUNTS} prompts, got {query_prompts}.")
    try:
        query_embedding = embed_sessions(request.sessions)
    except ValueError as exc:  # too few keystrokes (features.extract.MIN_KEYSTROKES)
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    similarity = cosine_similarity(query_embedding, np.array(entry["embedding"]))
    # The threshold depends on how steady both sides are: the claimed template (E prompts) and
    # this query (Q prompts).
    threshold, eer, extrapolated = auth_threshold(entry["enroll_prompts"], query_prompts)

    return AuthenticateResponse(
        person_id=entry["person_id"],
        name=entry["name"],
        accepted=similarity >= threshold,
        similarity=similarity,
        threshold=threshold,
        operating_point=OPERATING_POINT,
        enroll_prompts=entry["enroll_prompts"],
        query_prompts=query_prompts,
        eer_percent=eer,
        extrapolated=extrapolated,
    )
