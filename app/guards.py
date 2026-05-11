from dataclasses import dataclass


@dataclass
class GuardResult:
    blocked: bool
    reason: str = ""


BLOCK_PATTERNS = {
    "final underwriting decision": "I can’t provide final underwriting decisions.",
    "guaranteed premium": "I can only provide indicative premium estimates.",
    "medical diagnosis": "I can’t provide medical advice or diagnosis.",
}


def apply_guardrails(text: str) -> GuardResult:
    lower = text.lower()
    for pattern, reason in BLOCK_PATTERNS.items():
        if pattern in lower:
            return GuardResult(blocked=True, reason=reason)
    return GuardResult(blocked=False)
