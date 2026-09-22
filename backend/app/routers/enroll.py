"""POST /enroll -- extract features, embed, and store a new gallery entry."""

import numpy as np
from fastapi import APIRouter, HTTPException

from features.extract import extract_features

from ..embedding import embed
from ..gallery import add_entry
from ..schemas import EnrollRequest, EnrollResponse

router = APIRouter()


@router.post("/enroll", response_model=EnrollResponse)
def enroll(request: EnrollRequest) -> EnrollResponse:
    try:
        embeddings = []
        for session in request.sessions:
            events = [event.model_dump() for event in session.events]
            features = extract_features(events)
            embeddings.append(embed(features))
    except ValueError as exc:  # too few keystrokes (features.extract.MIN_KEYSTROKES)
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # Pool per-session embeddings (mean) into the single stored template, per
    # CLAUDE.md's enrollment flow.
    pooled = np.mean(np.stack(embeddings), axis=0).tolist()
    entry = add_entry(request.person_id, request.name, pooled)

    return EnrollResponse(
        person_id=entry["person_id"],
        name=entry["name"],
        enrolled_at=entry["enrolled_at"],
        num_sessions=len(request.sessions),
    )
