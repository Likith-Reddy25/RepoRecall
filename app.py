import streamlit as st
import time
from hybrid_retrieval import answer_code_question

# Page configuration
st.set_page_config(
    page_title="RepoRecall - Codebase AI",
    page_icon="⚡",
    layout="wide"
)

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.title("⚡ RepoRecall Ops")
    st.caption("Event-Driven Codebase RAG & Sync")
    st.success("🟢 Local RAG Engine Active")

    st.divider()
    
    st.subheader("💡 Sample Questions")
    sample_queries = [
        "Hey there",
        "How is HTTPException defined or handled?",
        "How does APIRouter handle include_router and prefixes?",
        "How does BackgroundTasks work in responses?"
    ]
    
    for q in sample_queries:
        if st.button(q, use_container_width=True):
            st.session_state["preset_query"] = q

    st.divider()
    st.caption("Hybrid Search: Dense (MiniLM) + BM25 + FlashRank Reranker")

# ----------------- MAIN CHAT UI -----------------
st.title("🤖 RepoRecall: Codebase Assistant")
st.markdown("Ask technical questions about the repository. Answers are retrieved via **Hybrid Search (Dense + BM25 + FlashRank)**.")

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

# Determine input (typed by user or clicked from sidebar preset)
user_prompt = st.chat_input("Ask a question about the codebase...")
if "preset_query" in st.session_state:
    user_prompt = st.session_state.pop("preset_query")

# Handle new user query
if user_prompt:
    # 1. Show user message
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # 2. Call retrieval & LLM directly
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Searching vectors & reranking snippets..."):
            start_time = time.time()
            try:
                # Direct Python call to your hybrid RAG pipeline!
                result = answer_code_question(user_prompt)
                elapsed = time.time() - start_time

                if isinstance(result, dict):
                    ans = result.get("answer", "")
                    sources = result.get("sources", [])
                else:
                    ans = str(result)
                    sources = []

                message_placeholder.markdown(ans)
                st.caption(f"⚡ Answered in {elapsed:.2f}s")
                
                if sources:
                    with st.expander("📁 Sources Cited"):
                        for src in sources:
                            st.code(src, language="text")

                # Save to message history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": ans,
                    "sources": sources
                })
            except Exception as e:
                message_placeholder.error(f"❌ Error: {str(e)}")