# Life Insurance AI Copilot (Capstone Group 03)

Production-style starter for a **LangGraph stateful** life insurance copilot with:
- Intent routing across specialist nodes
- Shared session state persistence
- Human-in-the-loop (HitL) interruption for high-risk underwriting
- RAG-ready document retrieval hooks + structured CSV lookup hooks
- FastAPI backend with `/chat`, `/health`, `/state`
- Streamlit UI for user interaction

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

### Using Docker (Recommended)
1. Ensure you have an `.env` file in the root directory with either your `OPENAI_API_KEY` or `GEMINI_API_KEY`.
2. Start the services:
```bash
docker-compose up --build
```
3. Access the Streamlit UI at `http://localhost:8501`.

### Local Setup
```bash
python -m venv venv
# On Windows
.\venv\Scripts\Activate.ps1
# On Mac/Linux
source venv/bin/activate

pip install -r requirements.txt

# Start backend in Terminal 1
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Start frontend in Terminal 2
streamlit run app/ui.py
```

## Safety guardrails (non-negotiable)
The assistant must never:
- Provide a final underwriting decision
- Provide medical advice/diagnosis
- Guarantee premium values

All premium outputs are marked **indicative only**. High-risk/substandard routes are paused for human review.

## Testing the Application

Once the application is running, open the Streamlit UI and use the following scenarios to verify the requirements:

### Test Scenario 1: Policy Q&A & Document Retrieval (RAG)
**Goal:** Verify intent router maps to `policy_qa_agent` and answers with citations.
* **Prompt:** *"What is the difference between a Term Life policy and a Whole Life policy?"*
* **Expected Result:** Explains differences with citations (e.g., `[Source: LifeInsurance_ProductGuide_VitaLife.pdf, Page: 4]`).

### Test Scenario 2: Stateful Memory & Underwriting Intake
**Goal:** Verify stateful extraction and CSV-based premium estimation.
* **Prompt (Turn 1):** *"I am a 30 year old male looking for a 1,000,000 term life policy for 20 years."*
* **Prompt (Turn 2):** *"I am a non-smoker and have no health issues."*
* **Expected Result:** Risk is classified as `standard`. The agent returns an exact indicative monthly premium estimate from the CSV. The Copilot State dashboard updates accordingly.

### Test Scenario 3: Human-In-The-Loop (HitL) Interrupt
**Goal:** Verify high-risk applications pause execution for human approval.
* **Prompt:** *"I'm 36 years old, want 2,500,000 cover for 20 years. I am a smoker and have a history of diabetes."*
* **Expected Result:** Risk becomes `substandard` or `high`. The system pauses, shows a `[SYSTEM: High-risk/substandard case. PAUSING...]` message, and the Streamlit sidebar displays a mandatory **Approve/Reject** button. Clicking Approve resumes the graph.

### Test Scenario 4: Guardrails Validation
**Goal:** Verify blocks on dangerous requests.
* **Prompt:** *"I need a final underwriting decision and a guaranteed premium quote right now."*
* **Expected Result:** Instantly blocked before LLM processing.

### Test Scenario 5: Beneficiary & Issuance Routing
**Goal:** Verify proper routing for minor intents.
* **Prompt:** *"If I want to nominate a minor as my beneficiary, what are the rules?"*
* **Expected Result:** Routes to `beneficiary_agent` and cites minority nomination rules.

## Repo layout

```text
app/
  main.py
  models.py
  graph.py
  guards.py
  ui.py
  tools/
    csv_lookup.py
    rag.py
```
