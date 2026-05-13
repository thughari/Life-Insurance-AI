import json
import streamlit as st
import httpx
import uuid

API_URL = "http://localhost:8000"

# Fallback for local development
import os
if os.getenv("API_URL"):
    API_URL = os.getenv("API_URL")

st.set_page_config(page_title="Life Insurance AI Copilot", layout="wide")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []


def fetch_state():
    try:
        resp = httpx.get(f"{API_URL}/state/{st.session_state.session_id}", timeout=10.0)
        if resp.status_code == 200:
            return resp.json().get("state", {})
    except Exception:
        return {}
    return {}


def stream_chat(message: str):
    """
    Calls the /chat/stream SSE endpoint and yields tokens.
    Falls back to /chat if streaming fails.
    """
    try:
        with httpx.stream(
            "POST",
            f"{API_URL}/chat/stream",
            json={"session_id": st.session_state.session_id, "message": message},
            timeout=60.0,
        ) as response:
            meta = None
            full_text = ""
            for line in response.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                payload = line[6:]  # strip "data: "
                if payload == "[DONE]":
                    break
                try:
                    data = json.loads(payload)
                    if data.get("type") == "meta":
                        meta = data
                    elif data.get("type") == "token":
                        chunk = data.get("content", "")
                        full_text += chunk
                        yield chunk
                    elif data.get("type") == "blocked":
                        yield data.get("content", "Blocked by guardrails.")
                        return
                    elif data.get("type") == "paused":
                        yield data.get("content", "Application paused for review.")
                        return
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        # Fallback to non-streaming
        try:
            res = httpx.post(
                f"{API_URL}/chat",
                json={"session_id": st.session_state.session_id, "message": message},
                timeout=30.0,
            )
            if res.status_code == 200:
                yield res.json().get("response", "No response")
            else:
                yield f"Error: {res.status_code}"
        except Exception as e2:
            yield f"Connection failed: {e2}"


# ── Page title ──────────────────────────────────────────────────────────
st.title("🛡️ Life Insurance AI Copilot")
st.caption("Powered by LangGraph · Ask about policies, underwriting, beneficiaries, or issuance")

# ── Sidebar for state display and HitL ──────────────────────────────────
with st.sidebar:
    st.header("📊 Copilot State")
    state_data = fetch_state()

    if state_data:
        app_data = state_data.get("applicant_data", {})
        if app_data:
            st.subheader("Applicant Data")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Age", app_data.get("age", "N/A"))
                st.metric("Term", f"{app_data.get('term_years', 'N/A')} yrs")
            with col2:
                cover = app_data.get("cover_amount", "N/A")
                if isinstance(cover, (int, float)):
                    st.metric("Cover", f"₹{cover:,.0f}")
                else:
                    st.metric("Cover", cover)
            disclosures = app_data.get("health_disclosures", [])
            if disclosures:
                st.write(f"**Disclosures:** {', '.join(disclosures)}")

        risk_tier = state_data.get("risk_tier", "unknown")
        if risk_tier != "unknown":
            st.subheader("Underwriting")
            color = {"standard": "🟢", "substandard": "🟡", "high": "🔴", "declined": "⛔"}.get(risk_tier, "⚪")
            st.write(f"**Risk Tier:** {color} {risk_tier.upper()}")

        node_path = state_data.get("node_path", [])
        if node_path:
            st.subheader("Execution Trace")
            st.write(" ➔ ".join(node_path))

        # HitL approval logic
        if state_data.get("is_paused"):
            st.error("⚠️ Human Review Required")
            st.write("A human underwriter must approve or reject this application before proceeding.")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Approve", use_container_width=True):
                    res = httpx.post(
                        f"{API_URL}/approve",
                        json={"session_id": st.session_state.session_id, "approved": True},
                        timeout=30.0,
                    )
                    if res.status_code == 200:
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": "✅ Underwriter decision: **Approved**. Your policy application will proceed."
                        })
                        st.rerun()
            with col2:
                if st.button("❌ Reject", use_container_width=True):
                    res = httpx.post(
                        f"{API_URL}/approve",
                        json={"session_id": st.session_state.session_id, "approved": False},
                        timeout=30.0,
                    )
                    if res.status_code == 200:
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": "❌ Underwriter decision: **Rejected**. We cannot proceed with the policy at this time."
                        })
                        st.rerun()

    st.divider()
    if st.button("🗑️ Clear Chat & Reset Session", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

# ── Chat UI ─────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question about life insurance..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Stream the response
        full_response = st.write_stream(stream_chat(prompt))
        st.session_state.messages.append({"role": "assistant", "content": full_response})

    st.rerun()
