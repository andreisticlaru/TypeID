"""GET /map -- interactive 2-D map of the gallery; GET /map/points -- the data behind it.

Layout: UMAP is fit once on the Aalto background people (ids starting with "aalto_"), then every
other enrolled person is *placed into that fixed layout*. Adding someone therefore never moves the
existing points. On the 500-person background, UMAP kept 95% trustworthiness and placing unseen
people held 94% (PCA: 72%; t-SNE scored 96% but can't place new points).

2-D distances are approximate (about half of a point's true nearest neighbours stay nearby), so each
point also carries its exact nearest neighbours by cosine in the full 128 dimensions.
"""

import threading
from pathlib import Path

import numpy as np
from fastapi import APIRouter
from fastapi.responses import FileResponse

from ..gallery import get_all_entries

router = APIRouter()

BACKGROUND_PREFIX = "aalto_"
MIN_BACKGROUND = 30  # below this UMAP has too little to fit; fall back to PCA
NEIGHBORS = 3
MAP_PAGE = Path(__file__).resolve().parent.parent / "static" / "map.html"

_fit_lock = threading.Lock()
_fitted: dict = {"ids": None, "reducer": None}


def _unit(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def _background_reducer(ids: tuple, vectors: np.ndarray):
    with _fit_lock:
        if _fitted["ids"] != ids:
            import umap  # heavy import (numba); only pay for it when the map is opened

            _fitted["reducer"] = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42).fit(vectors)
            _fitted["ids"] = ids
        return _fitted["reducer"]


def _layout(ids: list[str], units: np.ndarray) -> tuple[np.ndarray, str]:
    background = [i for i, pid in enumerate(ids) if pid.startswith(BACKGROUND_PREFIX)]
    if len(background) >= MIN_BACKGROUND:
        reducer = _background_reducer(tuple(ids[i] for i in background), units[background])
        xy = np.zeros((len(ids), 2))
        xy[background] = reducer.embedding_
        others = [i for i in range(len(ids)) if i not in set(background)]
        if others:
            xy[others] = reducer.transform(units[others])
        return xy, "UMAP"

    if len(ids) < 2:
        return np.zeros((len(ids), 2)), "none"
    centered = units - units.mean(axis=0)
    _, _, components = np.linalg.svd(centered, full_matrices=False)
    return centered @ components[:2].T, "PCA (fewer than 30 background people)"


@router.get("/map")
def map_page() -> FileResponse:
    return FileResponse(MAP_PAGE)


@router.get("/map/points")
def map_points() -> dict:
    entries = get_all_entries()
    if not entries:
        return {"method": "none", "points": []}

    ids = [e["person_id"] for e in entries]
    vectors = np.array([e["embedding"] for e in entries])
    units = _unit(vectors)
    xy, method = _layout(ids, units)

    similarity = units @ units.T
    np.fill_diagonal(similarity, -np.inf)
    nearest = np.argsort(-similarity, axis=1)[:, :NEIGHBORS]

    # Colour follows the person, never their rank: slots go by enrollment order.
    real = sorted((i for i, pid in enumerate(ids) if not pid.startswith(BACKGROUND_PREFIX)), key=lambda i: entries[i]["enrolled_at"])
    slot = {i: s for s, i in enumerate(real)}

    points = [
        {
            "id": ids[i],
            "name": entries[i]["name"],
            "real": i in slot,
            "slot": slot.get(i),
            "x": float(xy[i, 0]),
            "y": float(xy[i, 1]),
            "vector": [round(float(v), 4) for v in vectors[i]],
            "norm": round(float(np.linalg.norm(vectors[i])), 3),
            "enrolled_at": entries[i]["enrolled_at"],
            "neighbors": [
                {"id": ids[j], "name": entries[j]["name"], "real": j in slot, "cosine": round(float(similarity[i, j]), 3)}
                for j in nearest[i]
            ],
        }
        for i in range(len(ids))
    ]
    return {"method": method, "points": points}
