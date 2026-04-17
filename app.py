"""
app.py — Research Paper Q&A Agent (Streamlit UI)
=================================================
Run this with:  streamlit run app.py

This is the deployment file. It imports the agent from agent.py.
MODIFY: You can change UI text, colours, and sidebar content here.
"""

import streamlit as st
import uuid

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Research Paper Q&A Agent",
    page_icon="📄",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title("📄 Research Paper Q&A Agent")
st.caption(
    "-BUILD BY : ROHIT KASAUDHAN\n"
    "\nAsk anything about AI/ML research papers — architectures, methods, benchmarks, and more.\n"
    
)

# ── Load Agent (cached — only runs once) ──────────────────────────────────────
@st.cache_resource(show_spinner="⏳ Loading agent and knowledge base...")
def load_agent():
    from agent import build_agent, DOCUMENTS
    app, embedder, collection, llm = build_agent()
    return app, embedder, collection, llm, DOCUMENTS


# Try to load agent; show friendly error if API key is missing
try:
    app, embedder, collection, llm, DOCUMENTS = load_agent()
    st.success(f"✅ Knowledge base loaded — {collection.count()} research paper documents")
except ValueError as e:
    st.error(str(e))
    st.info(
        "**How to fix:** Open the `.env` file and replace `your_gemini_api_key_here` "
        "with your actual Gemini API key from https://aistudio.google.com/app/apikey"
    )
    st.stop()
except Exception as e:
    st.error(f"Failed to load agent: {e}")
    st.stop()

# ── Session State ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())[:8]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📚 About")
    st.write(
        "This agent answers questions about AI/ML research papers using "
        "a RAG pipeline (ChromaDB + Gemini-2.5-flash) with conversation memory "
        "and web search for recent papers."
    )
    st.write(f"🔑 Session ID: `{st.session_state.thread_id}`")
    st.divider()

    st.subheader("📖 Topics Covered")
    for doc in DOCUMENTS:
        st.write(f"• {doc['topic']}")

    st.divider()
    st.subheader("💡 Example Questions")
    examples = [
        "What is the Transformer architecture?",
        "How does BERT differ from GPT?",
        "Explain Retrieval-Augmented Generation",
        "What is RLHF and how does it work?",
        "What benchmarks are used to evaluate LLMs?",
        "How do diffusion models work?",
    ]
    for ex in examples:
        if st.button(ex, key=ex, use_container_width=True):
            st.session_state._prefill = ex

    st.divider()
    if st.button("🗑️ New Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())[:8]
        if "_prefill" in st.session_state:
            del st.session_state._prefill
        st.rerun()

# ── Display Chat History ──────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "meta" in msg:
            st.caption(msg["meta"])

# ── Handle sidebar example button prefill ────────────────────────────────────
prefill = st.session_state.pop("_prefill", None)

# ── Chat Input ────────────────────────────────────────────────────────────────
prompt = st.chat_input("Ask about a research paper or AI/ML concept...") or prefill

if prompt:
    # Show user message
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Run agent
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching knowledge base..."):
            try:
                config = {"configurable": {"thread_id": st.session_state.thread_id}}
                result = app.invoke({"question": prompt}, config=config)
                answer    = result.get("answer", "Sorry, I could not generate an answer.")
                faith     = result.get("faithfulness", 0.0)
                sources   = result.get("sources", [])
                route     = result.get("route", "retrieve")

            except Exception as e:
                answer  = f"Error: {str(e)}"
                faith   = 0.0
                sources = []
                route   = "error"

        st.write(answer)

        # Metadata caption
        if sources:
            src_str = " | ".join(f"`{s}`" for s in sources)
            faith_emoji = "✅" if faith >= 0.7 else "⚠️"
            st.caption(
                f"{faith_emoji} Faithfulness: **{faith:.2f}** &nbsp;·&nbsp; "
                f"Route: `{route}` &nbsp;·&nbsp; Sources: {src_str}"
            )

    meta = (
        f"Faithfulness: {faith:.2f} | Route: {route} | Sources: {', '.join(sources)}"
        if sources else f"Route: {route}"
    )
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "meta": meta}
    )
