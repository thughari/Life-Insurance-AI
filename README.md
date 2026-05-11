# Life Insurance AI Copilot (Capstone Group 03)

Production-style starter for a **LangGraph stateful** life insurance copilot with:
- Intent routing across specialist nodes
- Shared session state persistence
- Human-in-the-loop (HitL) interruption for high-risk underwriting
- RAG-ready document retrieval hooks + structured CSV lookup hooks
- FastAPI backend with `/chat`, `/health`, `/state`

## Architecture (target)

```text
Streamlit UI -> FastAPI -> LangGraph Workflow -> {FAISS Retriever + CSV Lookup Tools}
                                   |
                              MemorySaver
                                   |
                         Shared state across turns
```

### Graph Nodes
- `intent_router`
- `underwriting_agent`
- `policy_qa_agent`
- `beneficiary_agent`
- `issuance_agent`
- `human_review` (only for high-risk/substandard)

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Safety guardrails (non-negotiable)
The assistant must never:
- Provide a final underwriting decision
- Provide medical advice/diagnosis
- Guarantee premium values

All premium outputs are marked **indicative only**. High-risk/substandard routes are paused for human review.

## Repo layout

```text
app/
  main.py
  models.py
  graph.py
  guards.py
  tools/
    csv_lookup.py
    rag.py
```
