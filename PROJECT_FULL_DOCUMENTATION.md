# Life Insurance AI Copilot – Complete Project Documentation

## 1) What this document is
This file explains:
- Every file in the repository and why it exists.
- Every function/class in the Python app and why it exists.
- The end-to-end application flow (including asynchronous streaming).
- The full feature list and optimization techniques.
- The architectural justification behind the stack.

---

## 2) Repository-wide file inventory

### Root-level files
- `Dockerfile`  
  Container build definition for packaging the backend/frontend runtime and dependencies. Optimized for GCP Cloud Run.
- `docker-compose.yml`  
  Multi-service orchestration for running the app stack locally via Docker.
- `requirements.txt`  
  Highly optimized Python dependency lock/input for FastAPI, Streamlit, LangGraph, FAISS, PyMongo, etc. Extraneous libraries removed for minimum image size.
- `README.md`  
  Developer-facing quickstart, architecture overview, and test scenarios.
- `PROJECT_FULL_DOCUMENTATION.md`  
  (This file) Complete technical + functional documentation.

### Application code (`app/`)
- `app/main.py`  
  FastAPI service entrypoint. Exposes synchronous chat, SSE streaming chat, health checks, state retrieval, concurrent session listing, and session deletion.
- `app/graph.py`  
  LangGraph workflow definition. Contains all 9 specialist agent nodes, routing logic, LLM tool orchestration, and global In-Memory Caching setup.
- `app/models.py`  
  Pydantic state/request/response schemas, shared domain typing, and custom LangGraph reducers (e.g., context window optimization).
- `app/guards.py`  
  Non-negotiable guardrail block checks (Regex/Heuristic) before LLM execution.
- `app/ui.py`  
  Streamlit frontend: Contains the SSE stream listener, the multimodal file uploaders (Voice/Image), chat interface, and the sidebar session-management dashboard.

### Tool modules (`app/tools/`)
- `app/tools/rag.py`  
  PDF loading, chunking, HuggingFace local embedding, FAISS index build/load, and retrieval context assembly.
- `app/tools/csv_lookup.py`  
  Actuarial risk classification and indicative premium estimation from CSV reference tables.

### Data assets (`app/data/`)
- PDFs: `LifeInsurance_Glossary_VitaLife.pdf`, `PolicyTerms_Conditions_VitaLife.pdf`, etc. (8 source files for RAG).
- CSVs: `RiskScore_Classification_Table.csv`, `PremiumRate_ReferenceTable.csv` (2 source files for Underwriting).
- `faiss_index/` – Persisted vector index artifacts for instant retrieval.
- `sessions.json` – Local fallback storage for active sessions if MongoDB is not present.

---

## 3) Function-by-function explanation

## `app/main.py`
- `health()`  
  Lightweight liveness endpoint to confirm backend service availability.
- `chat(req: ChatRequest)`  
  Synchronous main orchestration endpoint: Invokes LangGraph, appends conversation history, returns full structured response.
- `chat_stream(req: ChatRequest)`  
  **Asynchronous SSE streaming endpoint**: 
  Executes `compiled_graph.astream_events` to yield tokens directly to the frontend in real-time. Emits `meta` chunks (for UI node traces) and `token` chunks (for text).
- `list_sessions()`  
  **Concurrent session fetching endpoint**: Iterates through active tracking IDs and uses `asyncio.gather(*tasks)` to hit MongoDB concurrently. Eliminates UI bottleneck.
- `delete_session(session_id: str)`  
  Removes a session from the `copilot_db.active_sessions` MongoDB collection, ensuring it vanishes from the UI.
- `approve(req: ApprovalRequest)`  
  Human-in-the-loop continuation endpoint for paused underwriting sessions. Stores approval decision and resumes the state graph.
- `get_state(session_id: str)`  
  Session state inspection endpoint. Directly queries the `MongoDBSaver` checkpointer to rebuild UI dashboards.

## `app/graph.py`
- `get_llm()`  
  Provider selector (Groq, Google Gemini, or OpenAI) based on `.env` keys.
- `intent_router(state: CopilotState) -> Dict`  
  Classifies user request into a domain intent using structured output schemas.
- `underwriting_agent(state: CopilotState) -> Dict`  
  Extracts applicant attributes, queries `csv_lookup`, computes indicative premium, flags human review requirements, and drafts responses.
- `policy_qa_agent`, `beneficiary_agent`, `issuance_agent`, `lapse_revival_agent`  
  Specialized RAG agents using targeted retrieval string logic to fetch relevant chunks from FAISS and answer with citations.
- `policy_comparison_agent(state: CopilotState) -> Dict` (Bonus)  
  Generates structured Markdown comparison tables for different policy types.
- `lapse_prediction_agent(state: CopilotState) -> Dict` (Bonus)  
  Analyzes simulated payment histories to predict lapse risk and suggest proactive interventions.
- `human_review(state: CopilotState) -> Dict`  
  Adds an explicit pause/review message when risk is elevated (HitL).
- `build_graph()`  
  Registers nodes/edges, configures the `MongoDBSaver` checkpointer using `pymongo.MongoClient`, and compiles interrupt behavior (`interrupt_before=["human_review"]`).

## `app/models.py`
- `ChatRequest`, `ChatResponse`  
  API data contracts.
- `CopilotState(TypedDict)`  
  Canonical cross-turn state model persisted in the graph checkpointer.
- `add_and_truncate_history(left, right)`  
  **Custom LangGraph Reducer Function**: Replaces standard `operator.add` for `conversation_history`. It strictly truncates the conversation array to the last 10 messages (5 pairs), protecting the LLM's context window from blowing up over long sessions.

## `app/guards.py`
- `apply_guardrails(text: str) -> GuardResult`  
  Regex/Keyword blocker for prohibited asks (final decision, guaranteed premium, prompt injection).

## `app/ui.py`
- `fetch_state()` / `stream_chat(prompt)`  
  HTTP client wrappers to interact with FastAPI. The `stream_chat` generator parses `Server-Sent Events (SSE)` using `httpx.stream`, yielding raw tokens to `st.write_stream`.
- **Sidebar Logic**:
  - Automatically fetches `/sessions` to populate a list of historic threads.
  - Generates UI buttons to `🔀 Switch to this session` and `🗑️ Delete Session`.
- **Multimodal Input Logic**:
  - Uses `st.file_uploader` to accept `.png`, `.jpg`, `.mp3`, `.wav`, `.m4a`.
  - Uses the `google.genai` SDK for image OCR extraction.
  - Uses the `openai.OpenAI` SDK (Whisper-1) for audio transcription.

## `app/tools/rag.py`
- `build_faiss_index()`  
  One-time PDF ingestion using `PyPDFLoader`, `RecursiveCharacterTextSplitter`, and `HuggingFaceEmbeddings` (local embedding model to save costs).
- `retrieve_policy_context(query: str) -> str`  
  Similarity search returning context blocks formatted with source document and page number markers.

## `app/tools/csv_lookup.py`
- `classify_risk(disclosures)` & `indicative_premium_lookup(age, cover, term, tier)`  
  Maps inputs via Pandas DataFrame filtering to output precise actuarial calculations, falling back to heuristic formulas if out of bounds.

---

## 4) End-to-end Streaming Application Flow

1. **User interacts via Streamlit** (`app/ui.py`): Types a message, speaks via mic, or uploads a document.
2. Multimodal inputs are pre-processed into text via Whisper/Gemini.
3. UI opens a long-lived `httpx.stream` POST request to `FastAPI /chat/stream`.
4. FastAPI invokes `apply_guardrails`. If safe, proceeds.
5. FastAPI executes `compiled_graph.astream_events()`.
6. LangGraph runs `intent_router`, branching to the appropriate Agent Node (e.g., `policy_qa_agent`).
7. **In-Memory Cache Check**: Langchain intercepts the LLM call. If the prompt was exactly answered before, it returns 0-latency cached tokens.
8. If not cached, the Agent queries RAG/CSV and streams tokens back to FastAPI.
9. FastAPI yields these tokens as `data: {"type": "token", "content": "..."}`.
10. Streamlit reads the stream and renders tokens live via `st.write_stream`.
11. **State Persistence**: Before FastAPI yields `[DONE]`, it calls `aupdate_state()` to save the turn into the MongoDB checkpoint database.
12. UI sidebar re-renders the `Node Execution Trace` and updating session data.
13. If `human_review` is reached, the graph pauses, state is saved, and UI presents ✅ Approve / ❌ Reject buttons. Clicking calls `POST /approve`, resuming the exact graph state from MongoDB!

---

## 5) Complete Feature & Optimization List

### Core Features
- **Stateful multi-turn sessions** keyed securely by UUIDs.
- **Intent routing** across 9 specialized agent subdomains.
- **Underwriting intake extraction** & **CSV-backed risk stratification**.
- **RAG policy Q&A** with Source/Page provenance citations.
- **Human-in-the-loop interruption** for elevated-risk underwriting.
- **Safety guardrails** against PII, prompt injection, and final decisions.

### Performance & Innovation Upgrades
- **MongoDB Checkpointing**: True session persistence allowing users to leave the app and return to their session hours later.
- **Concurrent API Fetching**: `asyncio.gather` reduces sidebar loading times from O(N) to O(1) latency.
- **In-Memory LLM Caching**: Identical queries bypass the LLM entirely, saving tokens and answering instantly.
- **Dynamic Context Optimizer**: Custom reducer logic strictly bounds the state's `conversation_history` to 10 messages, preventing Out-Of-Memory/Token-Limit failures on deep sessions.
- **SSE Token Streaming**: Eliminates perceived wait times by showing users text generation live.
- **Seamless Session Management**: UI to review past traces, resume contexts, and delete junk sessions.

---

## 6) Why this architecture exists

- **LangGraph over simple Chains**: Pure LangChain struggles with cyclic logic and explicit pauses. LangGraph's deterministic state-machine and interruptibility (`interrupt_before=["human_review"]`) makes HitL and complex conditional routing safe and predictable.
- **MongoDB (`MongoDBSaver`) over SQLite**: SQLite locks database files on concurrent writes, causing crashes in multi-user asynchronous environments like FastAPI/Cloud Run. MongoDB naturally handles highly concurrent read/writes for state checkpoints.
- **FastAPI SSE over WebSockets**: Server-Sent Events (SSE) operate natively over standard HTTP/1.1 or HTTP/2 without requiring complex WebSocket proxy configurations (which often fail or drop connections in Cloud Run and serverless environments).
- **Separation of Concerns**: FastAPI isolates the heavy orchestration, allowing the Streamlit UI to remain a thin, replaceable client. If you want to build a React Native mobile app tomorrow, the FastAPI backend requires zero changes.
- **Local HuggingFace Embeddings**: Using `sentence-transformers` locally avoids paying OpenAI/Google per-token for embedding 8 dense PDF documents, saving immense operating costs in production.
