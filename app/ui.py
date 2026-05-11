import streamlit as st
import httpx
import uuid

API_URL = "http://backend:8000"

st.set_page_config(page_title="Life Insurance AI Copilot", layout="wide")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

def fetch_state():
    try:
        resp = httpx.get(f"{API_URL}/state/{st.session_state.session_id}")
        if resp.status_code == 200:
            return resp.json().get("state", {})
    except Exception as e:
        return {}
    return {}

st.title("Life Insurance AI Copilot 🛡️")

# Sidebar for State display and HitL
with st.sidebar:
    st.header("Copilot State 📊")
    state_data = fetch_state()
    
    if state_data:
        app_data = state_data.get("applicant_data", {})
        st.subheader("Applicant Data")
        st.write(f"**Age**: {app_data.get('age', 'N/A')}")
        st.write(f"**Cover**: {app_data.get('cover_amount', 'N/A')}")
        st.write(f"**Term**: {app_data.get('term_years', 'N/A')}")
        st.write(f"**Disclosures**: {', '.join(app_data.get('health_disclosures', [])) or 'None'}")
        
        st.subheader("Underwriting")
        st.write(f"**Risk Tier**: {state_data.get('risk_tier', 'unknown')}")
        
        node_path = state_data.get("node_path", [])
        st.subheader("Execution Trace")
        st.write(" ➔ ".join(node_path) if node_path else "No nodes executed yet")
        
        # HitL approval logic based on backend state
        if state_data.get("is_paused"):
            st.error("⚠️ Human Review Required")
            if st.button("Approve Application"):
                res = httpx.post(f"{API_URL}/approve", json={"session_id": st.session_state.session_id, "approved": True})
                if res.status_code == 200:
                    st.session_state.messages.append({"role": "assistant", "content": "Underwriter decision: Approved. Your policy application will proceed."})
                    st.rerun()
            if st.button("Reject Application"):
                res = httpx.post(f"{API_URL}/approve", json={"session_id": st.session_state.session_id, "approved": False})
                if res.status_code == 200:
                    st.session_state.messages.append({"role": "assistant", "content": "Underwriter decision: Rejected. We cannot proceed with the policy."})
                    st.rerun()

    st.divider()
    if st.button("🗑️ Clear Chat & Reset Session"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

# Chat UI
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question about life insurance..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                res = httpx.post(
                    f"{API_URL}/chat", 
                    json={"session_id": st.session_state.session_id, "message": prompt},
                    timeout=30.0
                )
                if res.status_code == 200:
                    data = res.json()
                    response_text = data.get("response", "No response")
                    st.markdown(response_text)
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                else:
                    err_msg = f"Error: {res.status_code} {res.text}"
                    st.error(err_msg)
                    st.session_state.messages.append({"role": "assistant", "content": err_msg})
            except Exception as e:
                st.error(f"Connection failed: {e}")
                
    st.rerun()
