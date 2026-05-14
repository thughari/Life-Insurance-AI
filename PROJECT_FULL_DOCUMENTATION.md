# Life Insurance AI Copilot – Complete Project Documentation

## 1) Purpose of this document
This document is a code-aligned reference for the current implementation. It covers:
- every module and data file,
- every class/function in the Python codebase,
- end-to-end runtime flow,
- API contracts,
- guardrails and human-in-the-loop behavior.

> Scope note: this file documents code present as of this revision. Generated runtime artifacts (for example FAISS index files) are documented as operational outputs.

---

## 2) Repository inventory

### Root
- `README.md` – quickstart, architecture summary, endpoint overview.
- `PROJECT_FULL_DOCUMENTATION.md` – this full technical documentation.
- `problem-statement.txt` – capstone/project problem context.
- `requirements.txt` – Python dependencies.
- `Dockerfile` – container build definition.
- `docker-compose.yml` – local multi-service orchestration.
- `start.sh` – startup convenience script.
- `.gitignore`, `.gitkeep` – repository housekeeping.

### Application (`app/`)
- `app/main.py` – FastAPI app, REST endpoints, streaming endpoint, approval and state APIs.
- `app/graph.py` – LangGraph workflow, routing nodes, specialist agents, human-review node, streaming helper.
- `app/models.py` – request/response models and shared `CopilotState` typed state contract.
- `app/guards.py` – guardrail pattern matching for prohibited asks, injection attempts, and sensitive-data patterns.
- `app/ui.py` – Streamlit frontend with chat, streaming consumption, state dashboard, and approval actions.

### Tooling (`app/tools/`)
- `app/tools/rag.py` – PDF ingestion, embedding selection, FAISS index build/load, context retrieval.
- `app/tools/csv_lookup.py` – risk classification and indicative premium estimation from CSV references.

### Data (`app/data/`)
- Policy/underwriting source PDFs (RAG corpus).
- `RiskScore_Classification_Table.csv` – condition/risk mapping table.
- `PremiumRate_ReferenceTable.csv` – reference premium table.
- `faiss_index/` – generated FAISS persistence (built at runtime).

### Evaluation (`evaluation/`)
- `evaluation/test_set.json` – evaluation dataset.
- `evaluation/run_eval.py` – evaluator script calling backend and scoring outputs.
- `evaluation/eval_results.json` – generated evaluation output.

---

## 3) Code-level documentation (every class/function)

## `app/main.py`

### Globals
- `app = FastAPI(...)` – service instance.
- `compiled_graph = build_graph()` – compiled LangGraph workflow with memory checkpointing.

### Functions
- `health()`
  - **Route:** `GET /health`
  - **Purpose:** liveness probe.
  - **Return:** `{"status": "ok"}`.

- `chat(req: ChatRequest)`
  - **Route:** `POST /chat`
  - **Purpose:** synchronous chat orchestration.
  - **Flow:**
    1. Apply `apply_guardrails` on incoming message.
    2. If blocked, return `ChatResponse` with `node_path=["guardrail_block"]`.
    3. Check graph state for pending `human_review` interrupt.
    4. Invoke graph with `user_query` and `session_id` bound to LangGraph `thread_id`.
    5. Add pause warning message if graph is now interrupted before `human_review`.
    6. Append user/assistant turns using `compiled_graph.update_state` (reducer append semantics).
    7. Return `ChatResponse` with full state payload.

- `chat_stream(req: ChatRequest)`
  - **Route:** `POST /chat/stream`
  - **Purpose:** server-sent events (SSE) streaming interface.
  - **Behavior:**
    - Emits `blocked` event then `[DONE]` when guardrails block.
    - Emits `paused` event then `[DONE]` when session awaits human review.
    - Otherwise invokes graph first, then streams response text in token-like chunks (`chunk_size=8`) as `token` events plus initial `meta` event.
    - Appends pause message if interrupted after invoke.
    - Persists conversation history after stream completion.

- `approve(req: ApprovalRequest)`
  - **Route:** `POST /approve`
  - **Purpose:** resume interrupted high-risk underwriting flow with human decision.
  - **Validation:** rejects when no pending `human_review` interrupt exists.
  - **Actions:** updates graph state with `approved_by_human`, injects synthetic completion query, resumes graph via `invoke(None, ...)`.

- `get_state(session_id: str)`
  - **Route:** `GET /state/{session_id}`
  - **Purpose:** inspect current thread state for UI/debugging.
  - **Return:** state dict plus computed `is_paused` boolean.

### Classes
- `ApprovalRequest(BaseModel)`
  - Fields: `session_id: str`, `approved: bool`.

---

## `app/graph.py`

### LLM/provider helper
- `get_llm()`
  - Chooses provider in this order:
    1. Groq (`GROQ_API_KEY`) → `ChatGroq(model="llama-3.3-70b-versatile")`
    2. Google (`GOOGLE_API_KEY`/`GEMINI_API_KEY`) → `ChatGoogleGenerativeAI(model="gemini-2.5-flash")`
    3. OpenAI (`OPENAI_API_KEY`) → `ChatOpenAI(model="gpt-4o-mini")`
  - Raises if no key found.

### Structured schemas
- `IntentClassification(BaseModel)`
  - `intent`: one of `underwriting | policy_qa | beneficiary | issuance | lapse_revival`.

- `ApplicantDataExtract(BaseModel)`
  - Extracted underwriting fields:
    - `age: Optional[int]`
    - `cover_amount: Optional[int]`
    - `term_years: Optional[int]`
    - `health_disclosures: list[str]`

### Helpers
- `format_history(history: list) -> str`
  - Converts last 4 messages into prompt text; fallback when empty.

### Graph node functions
- `intent_router(state: CopilotState) -> Dict`
  - Intent classification using structured LLM output.
  - Fallback keyword router when LLM classification errors.
  - Appends `"intent_router"` to `node_path`.

- `underwriting_agent(state: CopilotState) -> Dict`
  - Extracts applicant data from current message.
  - Merges extracted fields with persisted `applicant_data`.
  - Computes `risk_tier = classify_risk(disclosures)`.
  - Computes `estimate = indicative_premium_lookup(...)`.
  - Flags `requires_human_review` for `high/substandard/declined`.
  - Generates safe natural-language response (no final decision).
  - Stores underwriting estimate under `node_outputs["underwriting"]`.

- `policy_qa_agent(state: CopilotState) -> Dict`
  - Builds search query (augments short follow-ups with prior assistant turn).
  - Retrieves context from RAG store.
  - Produces cited policy answer.

- `beneficiary_agent(state: CopilotState) -> Dict`
  - Retrieves beneficiary-focused context and answers nomination questions.

- `issuance_agent(state: CopilotState) -> Dict`
  - Retrieves issuance-focused context and answers on documents/timelines.

- `lapse_revival_agent(state: CopilotState) -> Dict`
  - Dedicated specialist for missed premium/lapse/revival/reinstatement guidance.

- `human_review(state: CopilotState) -> Dict`
  - Appends explicit pause/system warning content.

### Streaming helper
- `stream_agent_response(state: CopilotState, agent_name: str) -> AsyncIterator[str]`
  - Builds an agent-specific prompt and streams `AIMessageChunk` content.
  - Supports specialized prompts for each agent and generic fallback.
  - Returns async text chunks.

### Routing and graph assembly
- `route_from_intent(state: CopilotState) -> str`
  - Maps `state.intent` to node name (`underwriting_agent`, `policy_qa_agent`, `beneficiary_agent`, `issuance_agent`, `lapse_revival_agent`).

- `route_from_underwriting(state: CopilotState) -> str`
  - Sends to `human_review` if `requires_human_review=True`, else `END`.

- `build_graph()`
  - Builds `StateGraph(CopilotState)`.
  - Registers all nodes and conditional edges.
  - Adds `MemorySaver` checkpointer.
  - Compiles graph with `interrupt_before=["human_review"]`.

---

## `app/models.py`

### Type aliases
- `Intent` – literal union of supported intents.
- `RiskTier` – literal union `standard | substandard | high | declined | unknown`.

### API models
- `ChatRequest(BaseModel)`
  - `session_id: str` (non-empty)
  - `message: str` (non-empty)

- `ChatResponse(BaseModel)`
  - `session_id: str`
  - `node_path: List[str]`
  - `response: str`
  - `state: Dict[str, Any]`

### Shared LangGraph state
- `CopilotState(TypedDict, total=False)`
  - Core fields: `session_id`, `user_query`, `intent`, `response`, `applicant_data`, `risk_tier`, `policy_type_preference`, `node_outputs`, `requires_human_review`, `approved_by_human`.
  - Append-reducer fields:
    - `conversation_history: Annotated[List[Dict[str, str]], operator.add]`
    - `node_path: Annotated[List[str], operator.add]`

---

## `app/guards.py`

### Structures/constants
- `GuardResult` dataclass
  - `blocked: bool`, `reason: str`.

- `BLOCK_PATTERNS`
  - Direct phrase-based refusals for:
    - final underwriting decision requests,
    - guaranteed premium requests,
    - medical diagnosis/prescription requests.

- `INJECTION_PATTERNS`
  - Regex list for common jailbreak/prompt-injection attempts.

- `PHI_PATTERNS`
  - Regex list for sensitive information patterns (e.g., SSN/credit-card-style patterns and related terms).

### Function
- `apply_guardrails(text: str) -> GuardResult`
  - Executes in order:
    1. direct block phrases,
    2. injection regex detection,
    3. PHI/PII leakage detection,
    4. allow by default.

---

## `app/ui.py`

### Globals/session setup
- Resolves `API_URL` default (`http://backend:8000`) with environment override.
- Initializes Streamlit page config and per-session `session_id` + `messages` state.

### Functions
- `fetch_state()`
  - Calls backend `GET /state/{session_id}`.
  - Returns state dict or `{}` on failure.

- `stream_chat(message: str)`
  - Primary path: POST `/chat/stream`, parse SSE lines.
  - Handles event types: `meta`, `token`, `blocked`, `paused`, `[DONE]`.
  - Fallback path: POST `/chat` when streaming fails.

### UI workflow (top-level Streamlit script)
- Renders title/caption.
- Sidebar shows applicant data, risk tier, node path trace, pause status.
- Shows Approve/Reject actions when paused; calls `POST /approve`.
- Supports reset button to rotate `session_id` and clear transcript.
- Main chat pane displays history, sends new prompts, streams assistant response, stores messages, reruns.

---

## `app/tools/rag.py`

### Functions
- `get_embeddings()`
  - Embedding selection order:
    1. Groq key present → HuggingFace `all-MiniLM-L6-v2` embeddings,
    2. Google key present → Gemini embeddings (`models/gemini-embedding-001`),
    3. OpenAI key present → `OpenAIEmbeddings`.
  - Raises if no supported key present.

- `_current_provider() -> str`
  - Returns provider marker string (`huggingface`, `google`, `openai`, or `unknown`).

- `build_faiss_index(force: bool = False)`
  - Reuses existing index when provider marker matches.
  - Rebuilds when forced or provider changed / marker missing.
  - Loads all PDFs in `app/data`, splits into chunks (`1000/200`), builds FAISS, saves locally.
  - Writes provider marker file.

- `retrieve_policy_context(query: str, k: int = 3) -> str`
  - Ensures index exists.
  - Loads FAISS and runs similarity search.
  - Returns concatenated context blocks with `[Source, Page]` headers.

### Constants
- `FAISS_INDEX_PATH` – local vectorstore path.
- `PROVIDER_MARKER` – file storing embedding provider used for current index.

---

## `app/tools/csv_lookup.py`

### Constants
- `DATA_DIR`, `RISK_CSV_PATH`, `PREMIUM_CSV_PATH` – filesystem paths for lookup tables.

### Functions
- `classify_risk(disclosures: List[str]) -> str`
  - Defaults to `standard` when no disclosures.
  - Loads risk CSV and scans disclosure text against `specific_condition`.
  - Applies explicit smoker heuristic and tier mapping rules.
  - Returns highest-severity mapped tier among matches.

- `indicative_premium_lookup(age: int, cover_amount: int, term_years: int, risk_tier: str) -> Dict[str, str]`
  - Loads premium CSV.
  - Selects nearest available age/term/cover values.
  - Filters rows with default gender (`Male`) then picks price strategy by `risk_tier`.
  - Returns structured monthly estimate + disclaimer.
  - Uses formula fallback when table match is unavailable.

---

## `evaluation/run_eval.py`

### Functions
- `load_test_set()` – loads evaluation dataset JSON.
- `query_copilot(question: str, session_id: str = "eval-session") -> dict` – calls backend `/chat` for one prompt.
- `run_evaluation()` – iterates dataset, computes aggregate metrics, writes `evaluation/eval_results.json`.

---

## 4) API reference

- `GET /health`
  - Liveness endpoint.

- `POST /chat`
  - Body: `{"session_id": "...", "message": "..."}`
  - Returns `ChatResponse` model.

- `POST /chat/stream`
  - Body: same as `/chat`.
  - SSE events:
    - `meta` (node_path, pause state)
    - `token` (streamed text chunks)
    - `blocked` or `paused` terminal early messages
    - `[DONE]`

- `POST /approve`
  - Body: `{"session_id": "...", "approved": true|false}`
  - Resumes interrupted review flow.

- `GET /state/{session_id}`
  - Returns full persisted state with computed `is_paused` flag.

---

## 5) End-to-end execution flow

1. User enters prompt in Streamlit chat.
2. UI sends prompt to `/chat/stream` (fallback `/chat`).
3. Backend guardrails run first.
4. Backend checks if graph is currently paused at `human_review`.
5. Graph invocation begins with `intent_router`.
6. Query routes to specialist node.
7. Specialist node may call tools (RAG/CSV) and generate response.
8. Underwriting path may set `requires_human_review` and interrupt before `human_review`.
9. Backend/UI exposes pause state; underwriter can approve/reject via `/approve`.
10. Conversation history and node path accumulate in persisted thread state.

---

## 6) Functional capabilities checklist

- Intent classification and conditional routing.
- Multi-turn state persistence by `session_id`.
- Underwriting data extraction and state merge.
- CSV-based risk tiering.
- CSV-based indicative premium lookup with fallback formula.
- RAG retrieval across insurance PDFs with source/page markers.
- Specialist answering paths for policy, beneficiary, issuance, and lapse/revival.
- Human-in-the-loop pause/approve/reject workflow.
- Safety guardrails for prohibited content, injection, and sensitive data leakage.
- REST + SSE interfaces and Streamlit operator dashboard.
