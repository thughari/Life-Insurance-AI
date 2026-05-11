from typing import Dict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.models import CopilotState
from app.tools.csv_lookup import classify_risk, indicative_premium_lookup
from app.tools.rag import retrieve_policy_context


def intent_router(state: CopilotState) -> Dict:
    q = state.user_query.lower()
    if any(k in q for k in ["underwriting", "premium", "risk", "smoker", "diabetes"]):
        intent = "underwriting"
    elif any(k in q for k in ["beneficiary", "nominee", "share"]):
        intent = "beneficiary"
    elif any(k in q for k in ["issuance", "pending", "document", "timeline"]):
        intent = "issuance"
    else:
        intent = "policy_qa"
    return {"intent": intent, "node_path": state.node_path + ["intent_router"]}


def underwriting_agent(state: CopilotState) -> Dict:
    data = dict(state.applicant_data)
    disclosures = data.get("health_disclosures", ["smoker"])
    age = int(data.get("age", 35))
    cover = int(data.get("cover_amount", 1000000))
    term = int(data.get("term_years", 20))

    risk_tier = classify_risk(disclosures)
    estimate = indicative_premium_lookup(age, cover, term, risk_tier)

    requires_human = risk_tier in {"high", "substandard"}
    response = (
        f"Risk tier: {risk_tier}. Indicative monthly premium: {estimate['monthly_estimate']}. "
        f"{estimate['disclaimer']}"
    )
    return {
        "risk_tier": risk_tier,
        "requires_human_review": requires_human,
        "response": response,
        "node_outputs": {**state.node_outputs, "underwriting": {"estimate": estimate}},
        "node_path": state.node_path + ["underwriting_agent"],
    }


def policy_qa_agent(state: CopilotState) -> Dict:
    ctx = retrieve_policy_context(state.user_query)
    return {
        "response": f"Policy guidance: {ctx}",
        "node_path": state.node_path + ["policy_qa_agent"],
    }


def beneficiary_agent(state: CopilotState) -> Dict:
    return {
        "response": "Beneficiary nominations must total 100% shares; minor nominees require appointee/trustee details.",
        "node_path": state.node_path + ["beneficiary_agent"],
    }


def issuance_agent(state: CopilotState) -> Dict:
    return {
        "response": "Issuance requires KYC, medical records, and proposal verification. Typical turnaround is policy-dependent.",
        "node_path": state.node_path + ["issuance_agent"],
    }


def human_review(state: CopilotState) -> Dict:
    return {
        "response": state.response + " High-risk/substandard case routed for human underwriter review before continuation.",
        "approved_by_human": None,
        "node_path": state.node_path + ["human_review"],
    }


def route_from_intent(state: CopilotState) -> str:
    return state.intent or "policy_qa"


def route_from_underwriting(state: CopilotState) -> str:
    return "human_review" if state.requires_human_review else "end"


def build_graph():
    graph = StateGraph(CopilotState)

    graph.add_node("intent_router", intent_router)
    graph.add_node("underwriting_agent", underwriting_agent)
    graph.add_node("policy_qa_agent", policy_qa_agent)
    graph.add_node("beneficiary_agent", beneficiary_agent)
    graph.add_node("issuance_agent", issuance_agent)
    graph.add_node("human_review", human_review)

    graph.add_edge(START, "intent_router")
    graph.add_conditional_edges(
        "intent_router",
        route_from_intent,
        {
            "underwriting": "underwriting_agent",
            "policy_qa": "policy_qa_agent",
            "beneficiary": "beneficiary_agent",
            "issuance": "issuance_agent",
            "lapse_revival": "policy_qa_agent",
        },
    )
    graph.add_conditional_edges(
        "underwriting_agent",
        route_from_underwriting,
        {"human_review": "human_review", "end": END},
    )
    graph.add_edge("policy_qa_agent", END)
    graph.add_edge("beneficiary_agent", END)
    graph.add_edge("issuance_agent", END)
    graph.add_edge("human_review", END)

    return graph.compile(checkpointer=MemorySaver())
