"""POST /enroll -- extract features, embed, and store a new gallery entry."""

from fastapi import APIRouter, HTTPException

from ..embedding import embed_sessions
from ..gallery import add_entry, entry_exists
from ..schemas import PROMPT_COUNTS, EnrollRequest, EnrollResponse

router = APIRouter()


@router.post("/enroll", response_model=EnrollResponse)
def enroll(request: EnrollRequest) -> EnrollResponse:
    # A second "Alex" would otherwise silently overwrite the first Alex's template and report
    # success. Refuse unless the caller says outright that replacing is what they meant.
    if not request.replace and entry_exists(request.person_id):
        raise HTTPException(status_code=409, detail=f"{request.name} is already enrolled.")
    if len(request.sessions) not in PROMPT_COUNTS:
        raise HTTPException(status_code=422, detail=f"Enrol from {PROMPT_COUNTS} prompts, got {len(request.sessions)}.")

    try:
        template = embed_sessions(request.sessions)
    except ValueError as exc:  # too few keystrokes (features.extract.MIN_KEYSTROKES)
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    entry = add_entry(request.person_id, request.name, template.tolist(), len(request.sessions))

    return EnrollResponse(
        person_id=entry["person_id"],
        name=entry["name"],
        enrolled_at=entry["enrolled_at"],
        enroll_prompts=entry["enroll_prompts"],
    )
