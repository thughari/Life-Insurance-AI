import json
import asyncio
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.graph import build_graph, stream_agent_response
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
        if "human_review" in state_snapshot.next:
            return ChatResponse(
                session_id=req.session_id,
                node_path=["human_review_pending"],
                response="Your application is currently paused pending human underwriter review. Please wait for an underwriter to approve or reject it.",
                state=dict(state_snapshot.values) if state_snapshot.values else {},
            )

    # Invoke the LangGraph workflow
    result = compiled_graph.invoke(
        {"user_query": req.message, "session_id": req.session_id},
        config=config,
    )

    if not isinstance(result, dict):
        if hasattr(result, "model_dump"):
            result = result.model_dump()
        else:
            result = dict(result)

    # Check if interrupted after invoke
    state_snapshot = compiled_graph.get_state(config)
    is_paused = bool(state_snapshot and state_snapshot.next)

    response_text = result.get("response", "")
    if is_paused and "human_review" in (state_snapshot.next or []):
        response_text += "\n\n⚠️ **[SYSTEM: High-risk/substandard case. PAUSING for Human Underwriter review. Please approve or reject in the sidebar.]**"

    # Update conversation history in state (appended via reducer)
    compiled_graph.update_state(config, {
        "conversation_history": [
            {"role": "user", "content": req.message},
            {"role": "assistant", "content": response_text},
        ]
    })

    return ChatResponse(
        session_id=req.session_id,
        node_path=result.get("node_path", []),
        response=response_text,
        state=result,
    )


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """SSE streaming endpoint — streams LLM response tokens in real time."""

    guard = apply_guardrails(req.message)
    if guard.blocked:
        async def blocked_stream():
            yield f"data: {json.dumps({'type': 'blocked', 'content': guard.reason})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(blocked_stream(), media_type="text/event-stream")

    config = {"configurable": {"thread_id": req.session_id}}

    # Check if currently interrupted
    state_snapshot = compiled_graph.get_state(config)
    if state_snapshot and state_snapshot.next:
        if "human_review" in state_snapshot.next:
            async def paused_stream():
                yield f"data: {json.dumps({'type': 'paused', 'content': 'Your application is currently paused pending human underwriter review.'})}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(paused_stream(), media_type="text/event-stream")

    # Step 1: Run the graph synchronously to get the full result (routing + tool calls)
    result = await asyncio.to_thread(
        compiled_graph.invoke,
        {"user_query": req.message, "session_id": req.session_id},
        config,
    )

    if not isinstance(result, dict):
        result = dict(result) if not hasattr(result, "model_dump") else result.model_dump()

    # Check for HitL pause
    state_snapshot = compiled_graph.get_state(config)
    is_paused = bool(state_snapshot and state_snapshot.next)

    node_path = result.get("node_path", [])
    response_text = result.get("response", "")

    async def event_stream():
        # Send metadata first
        yield f"data: {json.dumps({'type': 'meta', 'node_path': node_path, 'is_paused': is_paused})}\n\n"

        # Stream response content
        # We chunk the pre-computed response to simulate streaming
        # (actual graph execution already happened — we stream the result)
        chunk_size = 8
        for i in range(0, len(response_text), chunk_size):
            chunk = response_text[i:i + chunk_size]
            yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"
            await asyncio.sleep(0.02)  # small delay for smooth streaming effect

        if is_paused:
            pause_msg = "\n\n⚠️ **[SYSTEM: High-risk/substandard case. PAUSING for Human Underwriter review.]**"
            yield f"data: {json.dumps({'type': 'token', 'content': pause_msg})}\n\n"
            response_with_pause = response_text + pause_msg
        else:
            response_with_pause = response_text

        yield "data: [DONE]\n\n"

        # Persist conversation history after streaming
        compiled_graph.update_state(config, {
            "conversation_history": [
                {"role": "user", "content": req.message},
                {"role": "assistant", "content": response_with_pause if is_paused else response_text},
            ]
        })

    return StreamingResponse(event_stream(), media_type="text/event-stream")


class ApprovalRequest(BaseModel):
    session_id: str
    approved: bool


@app.post("/approve")
def approve(req: ApprovalRequest):
    config = {"configurable": {"thread_id": req.session_id}}
    state_snapshot = compiled_graph.get_state(config)

    if not state_snapshot or "human_review" not in (state_snapshot.next or []):
        raise HTTPException(status_code=400, detail="No pending human review found for this session.")

    # Update state with approval and resume
    compiled_graph.update_state(config, {
        "approved_by_human": req.approved,
        "user_query": "Human Review Completed.",
        "response": f"Underwriter decision: {'✅ Approved' if req.approved else '❌ Rejected'}.",
    })

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
        data = dict(snapshot.values) if snapshot.values else {}

    data["is_paused"] = is_paused
    return {"session_id": session_id, "state": data}
