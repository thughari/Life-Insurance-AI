from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


Intent = Literal["underwriting", "policy_qa", "beneficiary", "issuance", "lapse_revival"]
RiskTier = Literal["standard", "substandard", "high", "declined", "unknown"]


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)


class ChatResponse(BaseModel):
    session_id: str
    node_path: List[str]
    response: str
    state: Dict[str, Any]


class CopilotState(BaseModel):
    session_id: str
    user_query: str = ""
    intent: Optional[Intent] = None
    response: str = ""
    applicant_data: Dict[str, Any] = Field(default_factory=dict)
    beneficiary_details: Dict[str, Any] = Field(default_factory=dict)
    risk_tier: RiskTier = "unknown"
    policy_type_preference: Optional[str] = None
    premium_estimate: Optional[Dict[str, Any]] = None
    conversation_history: List[Dict[str, str]] = Field(default_factory=list)
    node_outputs: Dict[str, Any] = Field(default_factory=dict)
    requires_human_review: bool = False
    approved_by_human: Optional[bool] = None
    node_path: List[str] = Field(default_factory=list)
