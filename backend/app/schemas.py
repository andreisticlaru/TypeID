"""Pydantic request/response models for the /enroll, /identify and /authenticate endpoints."""

from typing import Literal

from pydantic import BaseModel

# The only prompt counts thresholds are calibrated for (eval/calibrate_auth.py).
PROMPT_COUNTS = (1, 5, 10)


class KeystrokeEvent(BaseModel):
    key: str
    code: str
    type: Literal["keydown", "keyup"]
    t: float


class Session(BaseModel):
    sentence: str
    events: list[KeystrokeEvent]


class EnrollRequest(BaseModel):
    person_id: str
    name: str
    sessions: list[Session]  # one per transcribed prompt; the count must be in PROMPT_COUNTS
    replace: bool = False  # overwriting someone's template has to be asked for, never implied


class EnrollResponse(BaseModel):
    person_id: str
    name: str
    enrolled_at: str
    enroll_prompts: int


class IdentifyRequest(BaseModel):
    sessions: list[Session]  # one per prompt; the count must be in PROMPT_COUNTS


class IdentifyResult(BaseModel):
    person_id: str
    name: str
    similarity: float


class IdentifyResponse(BaseModel):
    results: list[IdentifyResult]
    matched: bool
    min_confidence: float  # the decision threshold, so the UI never hardcodes its own copy
    query_prompts: int
    found_percent: float  # how often an enrolled person is named correctly at this prompt count
    stranger_named_percent: float  # how often someone not enrolled is named anyway (the threshold's cap)


class Person(BaseModel):
    person_id: str
    name: str


class AuthenticateRequest(BaseModel):
    person_id: str  # the identity being CLAIMED; verification is 1:1, not a search
    sessions: list[Session]  # one per prompt; the count must be in PROMPT_COUNTS


class AuthenticateResponse(BaseModel):
    person_id: str
    name: str
    accepted: bool
    similarity: float
    threshold: float
    operating_point: str  # which calibrated point the threshold came from
    enroll_prompts: int
    query_prompts: int
    eer_percent: float  # measured error at this threshold, for this (enroll, query) pair
    extrapolated: bool  # no measured threshold for this pair; borrowed from the nearest one
