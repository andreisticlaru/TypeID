"""Decision thresholds per prompt count, at the equal-error operating point.

Measured on held-out subjects by: python -m eval.calibrate_auth --checkpoint model/encoder_hard.pt
(full output in eval/calibration.json). One global threshold per cell, never per person; see that
script's docstring for why.
"""

OPERATING_POINT = "EER"

# (enroll prompts, query prompts) -> (threshold, EER %). 1,000 held-out subjects per cell.
AUTH = {
    (1, 1): (0.370, 4.60),
    (1, 5): (0.491, 2.09),
    (1, 10): (0.506, 1.90),
    (5, 1): (0.497, 1.97),
    (5, 5): (0.714, 0.29),
    (5, 10): (0.768, 0.12),
    (10, 1): (0.514, 1.80),
    (10, 5): (0.761, 0.12),
}
# Aalto has at most 15 sessions per person, so 10 + 10 can't be measured without reusing
# enrolment sessions as queries. Borrow the nearest measured cell and say so in the response.
AUTH_FALLBACK = {(10, 10): (10, 5)}

# Open set, measured through the live pipeline on the live gallery (500 Aalto people enrolled from
# their first 5 sessions): enrolled people queried with LATER sessions, as real use always is, plus
# 600 strangers. Policy: names a stranger about 1 time in 5. Stricter caps looked right on paper (5%)
# but rejected most real attempts: an enrolled user's 5-prompt queries scored 0.81-0.94 against
# their profile while the best Aalto competitor never passed 0.794, and 0.858 turned half of them
# into "no match". In this demo nearly everyone who identifies has just enrolled, so a false
# "no match" is the common failure, and a near miss still shows the closest candidates.
# query prompts -> (threshold, % of enrolled people named correctly, % of strangers named).
# Enrolled people named as someone else: 10.7% / 4.7% / 2.0%.
IDENTIFY = {
    1: (0.75, 51.7, 21.8),
    5: (0.81, 88.3, 18.7),
    10: (0.82, 93.1, 17.8),
}


def auth_threshold(enroll_prompts: int, query_prompts: int) -> tuple[float, float, bool]:
    """(threshold, EER %, extrapolated) for a claim against an E-prompt template with a Q-prompt query."""
    cell = AUTH_FALLBACK.get((enroll_prompts, query_prompts), (enroll_prompts, query_prompts))
    threshold, eer = AUTH[cell]
    return threshold, eer, cell != (enroll_prompts, query_prompts)
