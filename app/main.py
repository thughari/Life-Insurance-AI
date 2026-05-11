from fastapi import FastAPI, HTTPException

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

    initial = CopilotState(session_id=req.session_id, user_query=req.message)
    result = compiled_graph.invoke(initial, config={"configurable": {"thread_id": req.session_id}})

    if not isinstance(result, dict):
        raise HTTPException(status_code=500, detail="Unexpected graph result")

    return ChatResponse(
        session_id=req.session_id,
        node_path=result.get("node_path", []),
        response=result.get("response", ""),
        state=result,
    )


@app.get("/state/{session_id}")
def get_state(session_id: str):
    snapshot = compiled_graph.get_state({"configurable": {"thread_id": session_id}})
    return {"session_id": session_id, "state": snapshot.values if snapshot else {}}
