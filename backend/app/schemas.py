"""Pydantic request/response models for the /enroll and /identify endpoints."""

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
