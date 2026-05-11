from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from app.graph import build_graph
from app.guards import apply_guardrails
from app.models import ChatRequest, ChatResponse, CopilotState

app = FastAPI(title="Life Insurance AI Copilot")
compiled_graph = build_graph()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    guard = apply_guardrails(req.message)
    if guard.blocked:
        return ChatResponse(
            session_id=req.session_id,
            node_path=["guardrail_block"],
            response=guard.reason,
            state={"guarded": True},
        )

    config = {"configurable": {"thread_id": req.session_id}}
    
    # Check if currently interrupted
    state_snapshot = compiled_graph.get_state(config)
    if state_snapshot and state_snapshot.next:
        # If the graph is paused at 'human_review', we shouldn't process normal chat.
        if "human_review" in state_snapshot.next:
            return ChatResponse(
                session_id=req.session_id,
                node_path=["human_review_pending"],
                response="Your application is currently paused pending human underwriter review. Please wait for an underwriter to approve it.",
                state=state_snapshot.values,
            )

    # Convert initial state to dict or object. Since CopilotState is Pydantic but we want dict update or object update
    # In langgraph with StateGraph(CopilotState), we pass a dict of updates.
    result = compiled_graph.invoke({"user_query": req.message, "session_id": req.session_id}, config=config)

    if not isinstance(result, dict):
        # Result might be a Pydantic object if CopilotState was returned directly.
        if hasattr(result, "model_dump"):
            result = result.model_dump()
        else:
            result = dict(result)

    # Check if interrupted after invoke
    state_snapshot = compiled_graph.get_state(config)
    is_paused = bool(state_snapshot and state_snapshot.next)
    
    response_text = result.get("response", "")
    if is_paused and "human_review" in state_snapshot.next:
        response_text += "\n\n[SYSTEM: High-risk/substandard case. PAUSING for Human Underwriter review. Please approve or reject.]"

    # Update conversation history in state
    history = result.get("conversation_history", [])
    history.append({"role": "user", "content": req.message})
    history.append({"role": "assistant", "content": response_text})
    compiled_graph.update_state(config, {"conversation_history": history})

    return ChatResponse(
        session_id=req.session_id,
        node_path=result.get("node_path", []),
        response=response_text,
        state=result,
    )


class ApprovalRequest(BaseModel):
    session_id: str
    approved: bool


@app.post("/approve")
def approve(req: ApprovalRequest):
    config = {"configurable": {"thread_id": req.session_id}}
    state_snapshot = compiled_graph.get_state(config)
    
    if not state_snapshot or "human_review" not in state_snapshot.next:
        raise HTTPException(status_code=400, detail="No pending human review found for this session.")
        
    # Update state with approval and resume
    compiled_graph.update_state(config, {"approved_by_human": req.approved, "user_query": "Human Review Completed.", "response": f"Underwriter decision: {'Approved' if req.approved else 'Rejected'}."})
    
    # Resume execution
    result = compiled_graph.invoke(None, config=config)
    
    if hasattr(result, "model_dump"):
        result = result.model_dump()
    elif not isinstance(result, dict):
        result = dict(result)

    return {"status": "resumed", "state": result}


@app.get("/state/{session_id}")
def get_state(session_id: str):
    snapshot = compiled_graph.get_state({"configurable": {"thread_id": session_id}})
    if not snapshot:
        return {"session_id": session_id, "state": {}}
        
    is_paused = bool(snapshot.next and "human_review" in snapshot.next)
    
    if hasattr(snapshot.values, "model_dump"):
        data = snapshot.values.model_dump()
    else:
        data = dict(snapshot.values)
        
    data["is_paused"] = is_paused
    return {"session_id": session_id, "state": data}
