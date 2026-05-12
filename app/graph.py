import os
from typing import Dict, Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.models import CopilotState
from app.tools.csv_lookup import classify_risk, indicative_premium_lookup
from app.tools.rag import retrieve_policy_context

def get_llm():
    if os.getenv("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini", temperature=0)
    elif os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    else:
        # For CI/CD or testing without key, though actual LLM calls will fail
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini", temperature=0)

class IntentClassification(BaseModel):
    intent: Literal["underwriting", "policy_qa", "beneficiary", "issuance", "lapse_revival"] = Field(
        description="Classify the user query into one of the allowed intents."
    )

def format_history(history: list) -> str:
    if not history: return "No previous conversation."
    return "\n".join([f"{msg['role']}: {msg['content']}" for msg in history[-4:]])

def intent_router(state: CopilotState) -> Dict:
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an intent classifier for a life insurance copilot. "
                   "Classify the user's intent based on their latest query and conversation history.\n"
                   "- underwriting: asking about risk, premium estimates, health disclosures, cover amount.\n"
                   "- policy_qa: asking about product types (term vs whole), coverages, riders.\n"
                   "- beneficiary: asking about nominations, nominees, shares.\n"
                   "- issuance: asking about pending documents, application status, issuance timelines.\n"
                   "- lapse_revival: asking about missed premiums, grace periods, reinstatement.\n"
                   "Default to policy_qa if unsure.\n\nConversation History:\n{history}"),
        ("user", "{query}")
    ])
    
    try:
        structured_llm = llm.with_structured_output(IntentClassification)
        result = structured_llm.invoke(prompt.format_prompt(history=format_history(state.conversation_history), query=state.user_query))
        intent = result.intent
    except Exception as e:
        print(f"LLM Routing failed: {e}")
        # fallback rule-based
        q = state.user_query.lower()
        if any(k in q for k in ["underwriting", "premium", "risk", "smoker", "diabetes", "cover", "age"]):
            intent = "underwriting"
        elif any(k in q for k in ["beneficiary", "nominee", "share"]):
            intent = "beneficiary"
        elif any(k in q for k in ["issuance", "pending", "document", "timeline", "status"]):
            intent = "issuance"
        else:
            intent = "policy_qa"
            
    return {"intent": intent, "node_path": state.node_path + ["intent_router"]}

from typing import Optional

class ApplicantDataExtract(BaseModel):
    age: Optional[int] = Field(default=None, description="Age of the applicant if mentioned.")
    cover_amount: Optional[int] = Field(default=None, description="Desired cover amount or sum assured if mentioned.")
    term_years: Optional[int] = Field(default=None, description="Policy term in years if mentioned.")
    health_disclosures: list[str] = Field(default_factory=list, description="Any health conditions, smoking habits, or lifestyle factors mentioned.")
    smoking_status: Optional[str] = Field(default=None, description="Smoking status if mentioned.")
    occupation: Optional[str] = Field(default=None, description="Occupation if mentioned.")
    annual_income: Optional[int] = Field(default=None, description="Annual income if mentioned.")

def underwriting_agent(state: CopilotState) -> Dict:
    llm = get_llm()
    
    # Extract data from current query
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract applicant data from the user query. Only extract what is explicitly mentioned."),
        ("user", "{query}")
    ])
    
    try:
        structured_llm = llm.with_structured_output(ApplicantDataExtract)
        extracted = structured_llm.invoke(prompt.format_prompt(query=state.user_query))
        
        # Merge with existing state
        data = dict(state.applicant_data)
        if extracted.age: data["age"] = extracted.age
        if extracted.cover_amount: data["cover_amount"] = extracted.cover_amount
        if extracted.term_years: data["term_years"] = extracted.term_years
        if extracted.health_disclosures:
            existing_health = data.get("health_disclosures", [])
            # avoid duplicates
            new_health = [d for d in extracted.health_disclosures if d not in existing_health]
            data["health_disclosures"] = existing_health + new_health
        if extracted.smoking_status:
            data["smoking_status"] = extracted.smoking_status
        if extracted.occupation:
            data["occupation"] = extracted.occupation
        if extracted.annual_income:
            data["annual_income"] = extracted.annual_income
    except Exception as e:
        print(f"Extraction failed: {e}")
        data = dict(state.applicant_data)

    disclosures = data.get("health_disclosures", [])
    age = int(data.get("age") or 35)
    cover = int(data.get("cover_amount") or 1000000)
    term = int(data.get("term_years") or 20)

    risk_tier = classify_risk(disclosures)
    estimate = indicative_premium_lookup(age, cover, term, risk_tier)

    requires_human = risk_tier in {"high", "substandard", "declined"}
    
    # Generate natural language response
    response_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an underwriting assistant. Summarize the applicant's current status and premium estimate. "
                   "Never provide a final underwriting decision. Emphasize estimates are non-binding. "
                   "If risk tier is high or substandard, mention it requires human review."),
        ("user", "Data: Age {age}, Cover {cover}, Term {term}, Conditions: {disclosures}.\n"
                 "Risk Tier: {risk_tier}. Estimate: {estimate_amt}\n"
                 "User query: {query}")
    ])
    
    resp_msg = llm.invoke(response_prompt.format_prompt(
        age=age, cover=cover, term=term, disclosures=disclosures, 
        risk_tier=risk_tier, estimate_amt=estimate['monthly_estimate'],
        query=state.user_query
    ))

    response_text = resp_msg.content.strip()
    response_text += "\n\nIndicative estimate only. Final premium subject to underwriting review."
    if requires_human:
        response_text += "\n\nThis case requires manual underwriting review before any eligibility decision can be made."
    
    return {
        "applicant_data": data,
        "risk_tier": risk_tier,
        "requires_human_review": requires_human,
        "response": response_text,
        "node_outputs": {**state.node_outputs, "underwriting": {"estimate": estimate}},
        "premium_estimate": estimate,
        "node_path": state.node_path + ["underwriting_agent"],
    }

def policy_qa_agent(state: CopilotState) -> Dict:
    llm = get_llm()
    
    # Improve RAG search query by prepending the last assistant response if query is too short/vague
    search_query = state.user_query
    if state.conversation_history and len(state.user_query.split()) < 6:
        search_query = f"{state.conversation_history[-1]['content']} {state.user_query}"
        
    ctx = retrieve_policy_context(search_query)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a Life Insurance Copilot answering policy questions. "
                   "Use the provided context to answer. Always include citations (Document and Page) from the context. "
                   "If the context does not contain the answer, say you don't know based on the documents.\n\n"
                   "Context:\n{context}\n\nConversation History:\n{history}"),
        ("user", "{query}")
    ])
    
    resp = llm.invoke(prompt.format_prompt(context=ctx, history=format_history(state.conversation_history), query=state.user_query))
    
    return {
        "response": resp.content,
        "node_path": state.node_path + ["policy_qa_agent"],
    }

def beneficiary_agent(state: CopilotState) -> Dict:
    llm = get_llm()
    ctx = retrieve_policy_context("Beneficiary nomination " + state.user_query)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a Beneficiary agent. Guide the user on nomination rules, share allocations, "
                   "and minor nominee rules based on the context. Include citations.\n\nContext:\n{context}"),
        ("user", "{query}")
    ])
    
    resp = llm.invoke(prompt.format_prompt(context=ctx, query=state.user_query))
    
    return {
        "response": resp.content,
        "node_path": state.node_path + ["beneficiary_agent"],
    }

def issuance_agent(state: CopilotState) -> Dict:
    llm = get_llm()
    ctx = retrieve_policy_context("Policy issuance status documents " + state.user_query)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an Issuance agent. Answer questions about pending documents and issuance timelines "
                   "based on the context. Include citations.\n\nContext:\n{context}"),
        ("user", "{query}")
    ])
    
    resp = llm.invoke(prompt.format_prompt(context=ctx, query=state.user_query))
    
    return {
        "response": resp.content,
        "node_path": state.node_path + ["issuance_agent"],
    }

def human_review(state: CopilotState) -> Dict:
    # In LangGraph, to actually pause, we can use the `interrupt` mechanism if configured.
    # We will mark it as paused in state so the API/UI knows.
    return {
        "response": state.response + "\n\n[SYSTEM: High-risk/substandard case. PAUSING for Human Underwriter review. Please approve or reject.]",
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

    # Use checkpointer. We can also add an interrupt_before=["human_review"] to physically halt the graph
    memory = MemorySaver()
    return graph.compile(checkpointer=memory, interrupt_before=["human_review"])
