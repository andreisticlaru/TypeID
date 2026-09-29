"""POST /identify -- rank gallery entries by similarity to a query embedding.

The ranking logic (rank_gallery) mirrors CLAUDE.md's identify() pseudocode.
Queries with too few keystrokes are rejected with HTTP 422, same as /enroll.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException

from ..embedding import embed_sessions
from ..gallery import get_all_entries
from ..schemas import PROMPT_COUNTS, IdentifyRequest, IdentifyResponse, IdentifyResult
from ..thresholds import IDENTIFY

router = APIRouter()

# Local record of live identify scores (names and similarities only, no text or timings), so the
# threshold can be checked against real browser typing rather than Aalto alone. Gitignored.
SCORE_LOG = Path(__file__).resolve().parents[2] / "identify_log.jsonl"
TOP_K = 10  # the UI shows the top 5 as the shortlist and 6-10 greyed out, for context


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def rank_gallery(
    query_embedding: np.ndarray,
    gallery: list[dict],
    min_confidence: float | None,
    top_k: int = TOP_K,
) -> tuple[list[IdentifyResult], bool]:
    """Rank every gallery entry by cosine similarity to a query embedding.

    Sorts all entries descending by similarity, returns the top-k, and flags
    "no confident match" (matched=False) if the gallery is empty or the best
    score falls below min_confidence -- the system declines to name anyone
    rather than force a top-1 guess. The candidates are returned either way:
    like a forensic examiner's shortlist, a near miss is information, and
    hiding it makes "no match" look like a malfunction.
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

    return results, matched


@router.get("/identify/rates")
def rates() -> dict:
    """Measured outcomes per query prompt count, so the UI can show the cost of each choice before typing."""
    return {
        "found_percent": {prompts: found for prompts, (_, found, _) in IDENTIFY.items()},
        "stranger_named_percent": {prompts: stranger for prompts, (_, _, stranger) in IDENTIFY.items()},
    }


@router.post("/identify", response_model=IdentifyResponse)
def identify(request: IdentifyRequest) -> IdentifyResponse:
    query_prompts = len(request.sessions)
    if query_prompts not in PROMPT_COUNTS:
        raise HTTPException(status_code=422, detail=f"Query with {PROMPT_COUNTS} prompts, got {query_prompts}.")
    try:
        query_embedding = embed_sessions(request.sessions)
    except ValueError as exc:  # too few keystrokes (features.extract.MIN_KEYSTROKES)
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # Calibrated against a gallery enrolled from 5 prompts, like the Aalto background. People
    # enrolled here from 1 or 10 prompts sit slightly off that scale; the background dominates.
    min_confidence, found, stranger = IDENTIFY[query_prompts]
    results, matched = rank_gallery(query_embedding, get_all_entries(), min_confidence)
    with open(SCORE_LOG, "a", encoding="utf-8") as log:
        log.write(json.dumps({
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "query_prompts": query_prompts,
            "threshold": min_confidence,
            "top": [[r.person_id, round(r.similarity, 4)] for r in results],
        }) + "\n")

    return IdentifyResponse(
        results=results,
        matched=matched,
        min_confidence=min_confidence,
        query_prompts=query_prompts,
        found_percent=found,
        stranger_named_percent=stranger,
    )
