from dataclasses import dataclass


@dataclass
class GuardResult:
    blocked: bool
    reason: str = ""


SECURITY_TEMPLATE = "I cannot assist with that request. Please consult a licensed insurance advisor or authorized representative."

BLOCK_KEYWORDS = [
    "am i approved",
    "final underwriting decision",
    "guaranteed premium",
    "exact premium",
    "medical diagnosis",
    "ignore instructions",
    "reveal system prompt",
    "system prompt",
    "internal data",
]


def apply_guardrails(text: str) -> GuardResult:
    lower = text.lower()
    for keyword in BLOCK_KEYWORDS:
        if keyword in lower:
            return GuardResult(blocked=True, reason=SECURITY_TEMPLATE)
    return GuardResult(blocked=False)
