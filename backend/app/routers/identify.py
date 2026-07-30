"""POST /identify -- rank gallery entries by similarity to a query embedding.

The ranking logic (rank_gallery) mirrors CLAUDE.md's identify() pseudocode
exactly and is real, complete code, not a stub -- it just isn't reachable
until extract_features()/embed() are implemented, since building the query
embedding needs both. Feature extraction/embedding failures are surfaced as
HTTP 501, same as /enroll.
"""

import numpy as np
from fastapi import APIRouter, HTTPException

from features.extract import extract_features

from ..embedding import embed
from ..gallery import get_all_entries
from ..schemas import IdentifyRequest, IdentifyResponse, IdentifyResult

router = APIRouter()

TOP_K = 5
# Placeholder threshold below which the best candidate is considered "no
# confident match" (CLAUDE.md's identify() pseudocode) -- tune once a trained
# model produces real similarity distributions.
MIN_CONFIDENCE = 0.5


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def rank_gallery(
    query_embedding: np.ndarray,
    gallery: list[dict],
    top_k: int = TOP_K,
    min_confidence: float | None = MIN_CONFIDENCE,
) -> tuple[list[IdentifyResult], bool]:
    """Rank every gallery entry by cosine similarity to a query embedding.

    Sorts all entries descending by similarity, returns the top-k, and flags
    "no confident match" (empty results, matched=False) if the gallery is
    empty or the best score falls below min_confidence -- mirroring the real
    forensic workflow of a human examiner reviewing candidates rather than a
    forced top-1 guess.
    """
    scored = [
        IdentifyResult(
            person_id=entry["person_id"],
            name=entry["name"],
            similarity=cosine_similarity(query_embedding, np.array(entry["embedding"])),
        )
        for entry in gallery
    ]
    scored.sort(key=lambda r: r.similarity, reverse=True)
    results = scored[:top_k]

    matched = bool(results) and (
        min_confidence is None or results[0].similarity >= min_confidence
    )
    if not matched:
        results = []

    return results, matched


@router.post("/identify", response_model=IdentifyResponse)
def identify(request: IdentifyRequest) -> IdentifyResponse:
    try:
        events = [event.model_dump() for event in request.events]
        features = extract_features(events)
        query_embedding = embed(features)
    except NotImplementedError as exc:
        raise HTTPException(
            status_code=501,
            detail="Feature extraction / embedding model not yet implemented",
        ) from exc

    gallery = get_all_entries()
    results, matched = rank_gallery(query_embedding, gallery)

    return IdentifyResponse(results=results, matched=matched)
