"""
Streamlit UI for the AI Learning & Study Assistant (RAG + Agent Tool Calling + Memory).
Run with: streamlit run app.py
"""
import os
import tempfile

import streamlit as st

import config
from rag_pipeline import load_or_create_vectorstore, add_files_to_vectorstore, get_retriever
from agent import build_agent_executor

st.set_page_config(page_title="StudyMate - AI Learning Assistant", page_icon="📚", layout="wide")


# ---------------- Session state bootstrapping ----------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = load_or_create_vectorstore()

if "kb_files" not in st.session_state:
    st.session_state.kb_files = []

if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role": "user"/"assistant", "content": str}]


def rebuild_agent():
    retriever = get_retriever(st.session_state.vectorstore)
    try:
        st.session_state.agent_executor = build_agent_executor(retriever)
        st.session_state.agent_error = None
    except Exception as e:
        st.session_state.agent_executor = None
        st.session_state.agent_error = str(e)


if "agent_executor" not in st.session_state:
    rebuild_agent()


def render_answer(text: str):
    """Render assistant output, giving quizzes a distinct card-like look."""
    if "QUIZ_START" in text and "QUIZ_END" in text:
        quiz_body = text.split("QUIZ_START", 1)[1].split("QUIZ_END", 1)[0].strip()
        st.markdown("#### 📝 Quiz")
        st.info(quiz_body)
    else:
        st.markdown(text)


# ---------------- Sidebar ----------------
with st.sidebar:
    st.title("📚 StudyMate")
    st.caption("RAG + Agent Tool Calling + Conversation Memory")

    st.subheader("1. Upload study material")
    uploaded_files = st.file_uploader(
        "Upload syllabus / notes (PDF, TXT, MD)",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )

    if st.button("Build / Update Knowledge Base", use_container_width=True):
        if not uploaded_files:
            st.warning("Upload at least one file first.")
        else:
            with st.spinner("Chunking and embedding documents..."):
                tmp_paths = []
                for f in uploaded_files:
                    tmp_dir = tempfile.mkdtemp()
                    path = os.path.join(tmp_dir, f.name)
                    with open(path, "wb") as out:
                        out.write(f.getbuffer())
                    tmp_paths.append(path)

                n_chunks = add_files_to_vectorstore(st.session_state.vectorstore, tmp_paths)
                st.session_state.kb_files.extend([f.name for f in uploaded_files])
                rebuild_agent()
            st.success(f"Indexed {n_chunks} chunks from {len(uploaded_files)} file(s).")

    if st.session_state.kb_files:
        st.subheader("Knowledge base contents")
        for name in st.session_state.kb_files:
            st.markdown(f"- 📄 {name}")

    st.divider()
    st.subheader("2. Or try the sample data")
    st.caption("Two ready-made files ship in /data for an instant demo.")
    if st.button("Load sample syllabus + notes", use_container_width=True):
        sample_paths = [
            os.path.join(config.DATA_DIR, "sample_syllabus.md"),
            os.path.join(config.DATA_DIR, "sample_notes.md"),
        ]
        sample_paths = [p for p in sample_paths if os.path.exists(p)]
        if sample_paths:
            with st.spinner("Indexing sample data..."):
                n_chunks = add_files_to_vectorstore(st.session_state.vectorstore, sample_paths)
                st.session_state.kb_files.extend([os.path.basename(p) for p in sample_paths])
                rebuild_agent()
            st.success(f"Indexed {n_chunks} chunks from sample data.")
        else:
            st.error("Sample files not found in ./data — check your working directory.")

    st.divider()
    if st.button("Clear chat memory", use_container_width=True):
        st.session_state.messages = []
        rebuild_agent()
        st.success("Conversation memory cleared.")

    st.divider()
    st.caption(f"Model: {config.GROQ_MODEL}")
    st.caption(f"Embeddings: {config.EMBEDDING_MODEL}")

    with st.expander("💡 Try asking"):
        st.markdown(
            "- What is the difference between BST and AVL trees?\n"
            "- Quiz me on graph traversal.\n"
            "- Make me a 5-day study plan for topics=BST|AVL Trees|BFS|DFS; days=5; hours_per_day=2"
        )


# ---------------- Main chat area ----------------
st.header("AI Learning & Study Assistant")
st.caption("Ask questions about your notes, request a quiz, or ask for a study schedule.")

if st.session_state.get("agent_error"):
    st.error(
        "Agent not initialized: "
        + st.session_state.agent_error
        + "\n\nSet GROQ_API_KEY in a .env file (copy .env.example) and restart the app."
    )

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        render_answer(msg["content"])

user_input = st.chat_input("Ask a question, e.g. 'Quiz me on Chapter 2' or 'Make a 5-day plan'")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        if st.session_state.agent_executor is None:
            answer = "The agent isn't configured yet — check the API key error above."
            st.markdown(answer)
        else:
            with st.spinner("Thinking..."):
                try:
                    result = st.session_state.agent_executor.invoke({"input": user_input})
                    answer = result.get("output", "I couldn't generate a response.")
                except Exception as e:
                    answer = f"Something went wrong: {e}"
            render_answer(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
