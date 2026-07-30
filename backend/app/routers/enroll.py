"""POST /enroll -- extract features, embed, and store a new gallery entry.

Feature extraction and embedding are both unimplemented stubs right now (they
depend on the Aalto dataset / trained model, which don't exist yet). This
endpoint stays honest about that: it returns HTTP 501 rather than faking a
successful enrollment, while the schema/routing/gallery-write plumbing is
fully wired up for when the real model lands.
"""

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
    except NotImplementedError as exc:
        raise HTTPException(
            status_code=501,
            detail="Feature extraction / embedding model not yet implemented",
        ) from exc

    # Pool per-session embeddings (mean) into the single stored template, per
    # CLAUDE.md's enrollment flow. Unreachable until embed() is implemented.
    pooled = np.mean(np.stack(embeddings), axis=0).tolist()
    entry = add_entry(request.person_id, request.name, pooled)

    return EnrollResponse(
        person_id=entry["person_id"],
        name=entry["name"],
        enrolled_at=entry["enrolled_at"],
        num_sessions=len(request.sessions),
    )
