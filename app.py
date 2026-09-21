import streamlit as st
import requests
import time

# Page configuration
st.set_page_config(
    page_title="RepoRecall - Codebase AI",
    page_icon="⚡",
    layout="wide"
)

API_BASE_URL = "http://localhost:8080"

# Custom Styling for modern dark look
st.markdown("""
<style>
    .reportview-container {
        margin-top: -2em;
    }
    .stChatFloatingInputContainer {
        bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.title("⚡ RepoRecall ")
    st.caption("Event-Driven Codebase RAG & Sync")
    
    # Check Backend Status
    try:
        res = requests.get(f"{API_BASE_URL}/docs", timeout=2)
        if res.status_code == 200:
            st.success("🟢 FastAPI Backend Online")
    except Exception:
        st.error("🔴 FastAPI Backend Offline (Run main.py)")

    st.divider()
    
    st.subheader("💡 Sample Questions")
    sample_queries = [
        "How is HTTPException defined or handled?",
        "How does APIRouter handle include_router and prefixes?",
        "How does BackgroundTasks work in responses?",
        "What are the core dependencies of this project?"
    ]
    
    for q in sample_queries:
        if st.button(q, use_container_width=True):
            st.session_state["preset_query"] = q

    st.divider()

    # Webhook Simulator Tool
    st.subheader("🛠️ Webhook Simulator")
    st.caption("Simulate a GitHub push commit event")
    with st.expander("Trigger Test Commit"):
        test_file = st.text_input("Modified File Path", value="fastapi/routing.py")
        if st.button("Simulate Push Webhook"):
            payload = {
                "ref": "refs/heads/main",
                "repository": {"full_name": "tiangolo/fastapi"},
                "commits": [
                    {
                        "id": "mock_commit_123",
                        "added": [],
                        "modified": [test_file],
                        "removed": []
                    }
                ]
            }
            try:
                wb_res = requests.post(f"{API_BASE_URL}/webhook", json=payload)
                if wb_res.status_code == 200:
                    st.toast(f"✅ Webhook triggered! Incremental sync queued for {test_file}", icon="🚀")
                else:
                    st.error(f"Webhook failed: {wb_res.status_code}")
            except Exception as e:
                st.error(f"Error: {e}")

# ----------------- MAIN CHAT UI -----------------
st.title("🤖 RepoRecall: Codebase Assistant")
st.markdown("Ask technical questions about the ingested repository. Answers are retrieved via **Hybrid Search (Dense + BM25 + FlashRank)**.")

# Initialize chat session history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant", 
            "content": "Hello! I'm your Codebase Assistant. Ask me anything about the codebase architecture, endpoints, or class implementations."
        }
    ]

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("📁 Sources Cited"):
                for src in msg["sources"]:
                    st.code(src, language="text")

# Determine input (either typed by user or clicked from sidebar preset)
user_prompt = st.chat_input("Ask a question about the codebase...")
if "preset_query" in st.session_state:
    user_prompt = st.session_state.pop("preset_query")

# Handle new user query
if user_prompt:
    # 1. Show user message
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # 2. Call FastAPI backend
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Searching vectors & reranking snippets..."):
            start_time = time.time()
            try:
                response = requests.post(
                    f"{API_BASE_URL}/chat", 
                    json={"query": user_prompt},
                    timeout=30
                )
                elapsed = time.time() - start_time

                if response.status_code == 200:
                    data = response.json()
                    answer_text = data.get("response", "")
                    
                    # If response is a dict with answer + sources
                    if isinstance(answer_text, dict):
                        ans = answer_text.get("answer", "")
                        sources = answer_text.get("sources", [])
                    else:
                        ans = str(answer_text)
                        sources = []

                    message_placeholder.markdown(ans)
                    st.caption(f"⚡ Retrieved and answered in {elapsed:.2f}s")
                    
                    if sources:
                        with st.expander("📁 Sources Cited"):
                            for src in sources:
                                st.code(src, language="text")

                    # Save to state
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": ans,
                        "sources": sources
                    })
                else:
                    error_msg = f"⚠️ Server Error ({response.status_code}): {response.text}"
                    message_placeholder.error(error_msg)
            except requests.exceptions.ConnectionError:
                message_placeholder.error("❌ Could not connect to FastAPI server. Make sure `python main.py` is running on port 8080.")
            except Exception as e:
                message_placeholder.error(f"❌ Error: {str(e)}")