"""Pydantic request/response models for the /enroll, /identify and /authenticate endpoints."""

from typing import Literal

from pydantic import BaseModel


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
    sessions: list[Session]  # 2-3 transcribed prompts, per CLAUDE.md enrollment flow
    replace: bool = False  # overwriting someone's template has to be asked for, never implied


class EnrollResponse(BaseModel):
    person_id: str
    name: str
    enrolled_at: str
    num_sessions: int


class IdentifyRequest(BaseModel):
    sentence: str
    events: list[KeystrokeEvent]


class IdentifyResult(BaseModel):
    person_id: str
    name: str
    similarity: float


class IdentifyResponse(BaseModel):
    results: list[IdentifyResult]
    matched: bool
    min_confidence: float  # the decision threshold, so the UI never hardcodes its own copy


class Person(BaseModel):
    person_id: str
    name: str


class AuthenticateRequest(BaseModel):
    person_id: str  # the identity being CLAIMED; verification is 1:1, not a search
    sentence: str
    events: list[KeystrokeEvent]


class AuthenticateResponse(BaseModel):
    person_id: str
    name: str
    accepted: bool
    similarity: float
    threshold: float
    operating_point: str  # which calibrated point the threshold came from
