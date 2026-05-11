from typing import Dict, List


RISK_KEYWORDS = {
    "diabetes": "substandard",
    "heart attack": "high",
    "cancer": "high",
    "smoker": "substandard",
}


def classify_risk(disclosures: List[str]) -> str:
    tier = "standard"
    for item in disclosures:
        keyword = item.lower().strip()
        mapped = RISK_KEYWORDS.get(keyword)
        if mapped == "high":
            return "high"
        if mapped == "substandard":
            tier = "substandard"
    return tier


def indicative_premium_lookup(age: int, cover_amount: int, term_years: int, risk_tier: str) -> Dict[str, str]:
    base = (cover_amount / 100000) * (age * 0.8 + term_years * 1.2)
    multiplier = {"standard": 1.0, "substandard": 1.35, "high": 1.75}.get(risk_tier, 1.0)
    monthly = round(base * multiplier, 2)
    return {
        "monthly_estimate": f"{monthly:.2f}",
        "disclaimer": "Indicative estimate only. Final premium is subject to underwriting review.",
    }
