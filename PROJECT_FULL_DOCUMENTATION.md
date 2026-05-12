# Life Insurance AI Copilot – Complete Project Documentation

## 1) What this document is
This file explains:
- Every file in the repository and why it exists.
- Every function/class in the Python app and why it exists.
- The end-to-end application flow.
- The full feature list.

---

## 2) Repository-wide file inventory

### Root-level files
- `Dockerfile`  
  Container build definition for packaging the backend/frontend runtime and dependencies.
- `docker-compose.yml`  
  Multi-service orchestration for running the app stack locally via Docker.
- `requirements.txt`  
  Python dependency lock/input for FastAPI, Streamlit, LangGraph/LangChain, FAISS, etc.
- `README.md`  
  Developer-facing quickstart, architecture overview, and test scenarios.
- `problem-statement.txt`  
  Assignment/capstone problem context and scope reference.
- `PROJECT_FULL_DOCUMENTATION.md`  
  (This file) complete technical + functional documentation.

### Application code (`app/`)
- `app/main.py`  
  FastAPI service entrypoint exposing chat, health, state, and human-approval endpoints.
- `app/graph.py`  
  LangGraph workflow definition, node logic, routing logic, and model/tool orchestration.
- `app/models.py`  
  Pydantic state/request/response schemas and shared domain typing.
- `app/guards.py`  
  Non-negotiable guardrail block checks before LLM execution.
- `app/ui.py`  
  Streamlit frontend: chat interface + state dashboard + human review actions.

### Tool modules (`app/tools/`)
- `app/tools/rag.py`  
  PDF loading, chunking, embedding, FAISS index build/load, and retrieval context assembly.
- `app/tools/csv_lookup.py`  
  Risk classification and indicative premium estimation from CSV reference tables.

### Data assets (`app/data/`)
- `LifeInsurance_Glossary_VitaLife.pdf` – domain glossary source for RAG context.
- `PolicyTerms_Conditions_VitaLife.pdf` – policy terms source for RAG.
- `LifeInsurance_ProductGuide_VitaLife.pdf` – product coverage source for RAG.
- `BeneficiaryNomination_Guidelines_VitaLife.pdf` – nomination rules source.
- `PolicyIssuance_Checklist_VitaLife.pdf` – issuance process source.
- `Underwriting_Guidelines_VitaLife.pdf` – underwriting source.
- `Lapse_Revival_Reinstatement_VitaLife.pdf` – lapse/reinstatement source.
- `Riders_AddOnBenefits_VitaLife.pdf` – rider/benefits source.
- `RiskScore_Classification_Table.csv` – condition-to-risk mapping input for underwriting risk tiering.
- `PremiumRate_ReferenceTable.csv` – reference premium table for indicative premium lookup.
- `faiss_index/index.faiss`, `faiss_index/index.pkl` – persisted vector index artifacts for retrieval.

---

## 3) Function-by-function explanation

## `app/main.py`
- `health()`  
  Lightweight liveness endpoint to confirm backend service availability.
- `chat(req: ChatRequest)`  
  Main orchestration endpoint:
  1) applies guardrails, 2) checks paused human-review state, 3) invokes LangGraph, 4) appends conversation history, 5) returns structured response.
- `approve(req: ApprovalRequest)`  
  Human-in-the-loop continuation endpoint for paused underwriting sessions; stores approval decision and resumes graph execution.
- `get_state(session_id: str)`  
  Session state inspection endpoint used by UI sidebar for live status, trace, and pause state.

Classes:
- `ApprovalRequest(BaseModel)`  
  Request schema for underwriter decision payload (`session_id`, `approved`).

## `app/graph.py`
- `get_llm()`  
  Provider selector that chooses OpenAI or Gemini model based on environment keys (with fallback).
- `format_history(history: list) -> str`  
  Compacts recent conversation messages into prompt-ready text.
- `intent_router(state: CopilotState) -> Dict`  
  Classifies latest user request into domain intent and records route path.
- `underwriting_agent(state: CopilotState) -> Dict`  
  Extracts applicant attributes, merges persistent state, classifies risk tier, computes indicative premium, flags human review requirements, and drafts underwriting-safe response.
- `policy_qa_agent(state: CopilotState) -> Dict`  
  Retrieves policy-document context and answers policy questions with citation-oriented prompting.
- `beneficiary_agent(state: CopilotState) -> Dict`  
  Specialized beneficiary nomination Q&A using targeted retrieval context.
- `issuance_agent(state: CopilotState) -> Dict`  
  Specialized policy issuance and pending-document Q&A using targeted retrieval context.
- `human_review(state: CopilotState) -> Dict`  
  Adds an explicit pause/review message in state when risk is elevated.
- `route_from_intent(state: CopilotState) -> str`  
  Conditional edge router from intent node to the relevant specialist node.
- `route_from_underwriting(state: CopilotState) -> str`  
  Conditional edge router from underwriting to either end or human review.
- `build_graph()`  
  Registers nodes/edges, enables state checkpointing, and compiles interrupt behavior (`interrupt_before=["human_review"]`).

Classes:
- `IntentClassification(BaseModel)`  
  Structured output schema for intent classification.
- `ApplicantDataExtract(BaseModel)`  
  Structured extraction schema for age/cover/term/health disclosures.

## `app/models.py`
- `ChatRequest(BaseModel)`  
  Input schema for `/chat` (session + message).
- `ChatResponse(BaseModel)`  
  Output schema for `/chat` (session + node trace + response + state snapshot).
- `CopilotState(BaseModel)`  
  Canonical cross-turn state model persisted in graph checkpointer.

Aliases:
- `Intent`  
  Restricted set of valid intents.
- `RiskTier`  
  Restricted set of valid risk tiers.

## `app/guards.py`
- `apply_guardrails(text: str) -> GuardResult`  
  Pattern blocker for prohibited asks (final decision, guaranteed premium, diagnosis), returning allow/block verdict.

Classes/constants:
- `GuardResult`  
  Structured guard outcome.
- `BLOCK_PATTERNS`  
  Phrase-to-reason mapping used for safety enforcement.

## `app/ui.py`
- `fetch_state()`  
  Helper that calls backend `/state/{session_id}` and returns current state for sidebar rendering.

Top-level UI flow exists as script logic (Streamlit pattern):
- initializes session/message state,
- renders sidebar diagnostics,
- renders approve/reject buttons when paused,
- renders chat transcript,
- sends new prompt to `/chat`,
- appends assistant result and reruns.

## `app/tools/rag.py`
- `get_embeddings()`  
  Embedding provider selector (OpenAI/Gemini/fallback fake embedding).
- `build_faiss_index()`  
  One-time PDF ingestion + splitting + vector index creation/persistence.
- `retrieve_policy_context(query: str, k: int = 3) -> str`  
  Similarity search and context formatter with source/page markers for downstream citation.

Constants:
- `FAISS_INDEX_PATH`  
  Filesystem location for persisted vector index.

## `app/tools/csv_lookup.py`
- `classify_risk(disclosures: List[str]) -> str`  
  Maps health/lifestyle disclosures to a normalized risk tier using CSV + fallback keyword rules.
- `indicative_premium_lookup(age, cover_amount, term_years, risk_tier) -> Dict[str, str]`  
  Finds closest premium-table match and returns indicative monthly premium + disclaimer, with formula fallback.

Constants:
- `DATA_DIR`, `RISK_CSV_PATH`, `PREMIUM_CSV_PATH`  
  Paths to underwriting reference datasets.

---

## 4) End-to-end app flow

1. **User types in Streamlit chat** (`app/ui.py`).
2. UI sends request to `POST /chat` on FastAPI (`app/main.py`).
3. Backend runs `apply_guardrails`; if blocked, returns immediate safe refusal.
4. Backend checks graph checkpoint state for pending human review pause.
5. If not paused, backend invokes compiled LangGraph with current query + session thread id.
6. Graph starts at `intent_router`.
7. Router sends query to one specialist path:
   - `underwriting_agent`
   - `policy_qa_agent`
   - `beneficiary_agent`
   - `issuance_agent`
   - (`lapse_revival` currently mapped to `policy_qa_agent`)
8. **Underwriting path** additionally:
   - extracts applicant entities,
   - updates stateful applicant profile,
   - computes risk + premium estimate from CSV tools,
   - if risk is high/substandard/declined, triggers human-review interrupt.
9. Graph returns response + updated state.
10. Backend appends user/assistant turns to `conversation_history` and persists to graph state.
11. UI renders assistant output + updated sidebar state.
12. If paused, UI shows **Approve/Reject** buttons, calling `POST /approve` to resume execution.

---

## 5) Complete feature list

- **Stateful multi-turn sessions** keyed by `session_id`.
- **Intent routing** across specialized insurance subdomains.
- **Underwriting intake extraction** (age, cover, term, disclosures).
- **CSV-backed risk stratification**.
- **CSV-backed indicative premium lookup** with numeric fallback.
- **RAG policy Q&A** across multiple policy PDFs.
- **Source/page provenance formatting** in retrieval context.
- **Human-in-the-loop interruption** for elevated-risk underwriting.
- **Manual approval/rejection resume flow** via API and UI.
- **Safety guardrails** against prohibited outputs.
- **Execution trace visibility** via `node_path` and sidebar state.
- **Health and state introspection endpoints** for operability/debugging.
- **Dockerized deployment path** with compose orchestration.

---

## 6) Why this architecture exists

- **LangGraph state model** enables deterministic orchestration and persistent conversation/application state.
- **Specialist node split** keeps prompts and responsibilities scoped, improving maintainability.
- **Tool separation (`rag.py`, `csv_lookup.py`)** cleanly isolates data retrieval/calculation logic from orchestration.
- **FastAPI + Streamlit split** allows simple service boundary: API first, UI as replaceable client.
- **HitL pause pattern** enforces compliance-like gating where automation should not finalize high-risk outcomes.
- **Guardrails before graph invoke** reduce unsafe output risk and wasted model/tool calls.
