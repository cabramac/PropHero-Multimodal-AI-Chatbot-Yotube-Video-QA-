import os
from typing import List, Dict, Any

import streamlit as st

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

try:
    from langchain.retrievers.multi_query import MultiQueryRetriever
except ImportError:
    from langchain_community.retrievers import MultiQueryRetriever


# 1. Streamlit basic config
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="PropHero AI Assistant",
    page_icon="🏠",
    layout="wide",
)

st.markdown("""
<style>

    /* REMOVE the red focus outline (TRUE SOURCE OF THE RED BORDER) */
    .stChatFloatingInputContainer *:focus {
        outline: none !important;
        box-shadow: none !important;
    }

    /* Disable browser default focus ring */
    *:focus-visible {
        outline: none !important;
    }

</style>
""", unsafe_allow_html=True)

# 2. Cached loaders: vectorstore + QA chain
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading vector store…")
def load_vectorstore() -> Chroma:
    """
    Load the persistent Chroma DB you created in Notebook 1/2.

    It assumes:
      persist_directory="data/chroma_db"
      collection_name="prophero_knowledge"
      embeddings: sentence-transformers/all-MiniLM-L6-v2
    """
    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = Chroma(
        collection_name="prophero_knowledge",
        persist_directory="data/chroma_db",
        embedding_function=embedding_function,
    )
    return vectordb


@st.cache_resource(show_spinner="Warming up PropHero assistant…")
def load_qa_chain() -> RetrievalQA:
    """
    Build the full RAG chain:
      Chroma → similarity retriever → MultiQuery retriever → LLM (OpenAI) → QA chain.
    """
    vectordb = load_vectorstore()

    llm = ChatOpenAI(
        model="gpt-4o-mini", 
        temperature=0.1,
    )

    base_retriever = vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10},
    )

    multiquery_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm,
        include_original=True,
    )

    qa_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are a helpful and friendly AI assistant for PropHero, a smart "
            "digital property investment platform.\n\n"
            "Use ONLY the information in the context below to answer the user's "
            "question in a clear, natural, and professional way.\n"
            "If the answer is not found in the context, say something like:\n"
            "\"I'm sorry, but I couldn't find information about that in PropHero's "
            "materials.\"\n\n"
            "Context:\n{context}\n\n"
            "Question:\n{question}\n\n"
            "Answer:\n"
        ),
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=multiquery_retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": qa_prompt},
    )

    return qa_chain


qa_chain = load_qa_chain()

# 3. Session state for chat history
# -----------------------------------------------------------------------------
if "projects" not in st.session_state:
    st.session_state["projects"] = {
        "Start your new investment": {
            "chats": {"Chat 1": []},
            "current_chat": "Chat 1",
        }
    }

# Select default project
if "current_project" not in st.session_state:
    st.session_state["current_project"] = "Start your new investment"


# 4. Helper: pretty rendering of sources
# -----------------------------------------------------------------------------
def render_sources(source_docs: List[Any]) -> None:
    """Render retrieved documents nicely in an expander."""
    if not source_docs:
        st.write("No sources returned.")
        return

    for i, doc in enumerate(source_docs, start=1):
        meta = doc.metadata or {}
        title = meta.get("title", "Untitled chunk")
        url = meta.get("url", "")
        source_type = meta.get("source_type", "unknown")
        video_id = meta.get("video_id", "")
        chunk_id = meta.get("chunk_id", "")

        with st.container(border=True):
            st.markdown(f"**Source {i}**  ·  _{source_type}_")
            st.markdown(f"**Title:** {title}")

            if url:
                st.markdown(f"[Open source]({url})")

            if video_id and source_type == "video":
                st.caption(f"Video id: `{video_id}`")

            if chunk_id:
                st.caption(f"Chunk id: `{chunk_id}`")


# 5. Sidebar – project info & controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("prophero_logo.png", width=160)

    # === 1) PROJECTS LIST ===
    st.markdown("### 📁 Projects")

    # Handle project creation
    new_project = st.text_input("Add new project")
    if st.button("Add project"):
        if new_project.strip() != "":
            st.session_state["projects"][new_project] = {
                "chats": {"Chat 1": []},
                "current_chat": "Chat 1",
            }
            st.session_state["current_project"] = new_project
            st.rerun()

    # Display projects as clickable
    for project_name in st.session_state["projects"].keys():
        if st.button(project_name):
            st.session_state["current_project"] = project_name
            st.rerun()

    st.markdown("---")

    # === 2) CHATS INSIDE CURRENT PROJECT ===
    st.markdown("### 🗂️ Chats")

    current_project = st.session_state["current_project"]
    chats = st.session_state["projects"][current_project]["chats"]

    # Button to create a new chat
    if st.button("➕ New Chat"):
        new_chat_name = f"Chat {len(chats) + 1}"
        chats[new_chat_name] = []
        st.session_state["projects"][current_project]["current_chat"] = new_chat_name
        st.rerun()

    # List all chats inside active project
    for chat_name in chats:
        if st.button(chat_name):
            st.session_state["projects"][current_project]["current_chat"] = chat_name
            st.rerun()



# 6. Main chat UI
# -----------------------------------------------------------------------------
left_col, mid_col, right_col = st.columns([1, 8, 1])
with left_col:
    st.image("prophero_logo.png", width=120)

st.title("PropHero AI Chatbot")
st.write("Ask questions about PropHero and property investing.")

# === LOAD CURRENT CHAT MESSAGES ===
current_project = st.session_state["current_project"]
current_chat = st.session_state["projects"][current_project]["current_chat"]

messages = st.session_state["projects"][current_project]["chats"][current_chat]

# Display chat history
for msg in messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input (bottom text box)
user_input = st.chat_input("Type your question here…")

if user_input:
    # 1) Add user message to history
    messages.append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2) Assistant "typing" indicator
with st.chat_message("assistant"):
    placeholder = st.empty()

    # If user has not asked anything yet → show friendly welcome
    if not user_input:
        placeholder.markdown("👋 I'm here and ready whenever you are!")
    else:
        placeholder.markdown("_Thinking…_ 🤔")

    # === 1) Build conversational memory (last 6 messages) ===
    current_project = st.session_state["current_project"]
current_chat = st.session_state["projects"][current_project]["current_chat"]
messages = st.session_state["projects"][current_project]["chats"][current_chat]

history_messages = []
for msg in messages[-6:]:
    if msg.get("content") is None:
        continue
    role = "User" if msg["role"] == "user" else "Assistant"
    history_messages.append(f"{role}: {msg['content']}")

    role = "User" if msg["role"] == "user" else "Assistant"
    history_messages.append(f"{role}: {msg['content']}")

    history_text = "\n".join(history_messages)

    # Prevent crash if there's a rerun before user types
    if not user_input:
        user_input = ""

    # === 2) Combine memory + new user query safely ===
    query_with_memory = (
        (history_text + "\n") if history_text else ""
    ) + f"User: {user_input}\nAssistant:"

    # === 3) Call RAG chain with memory ===
    try:
        result = qa_chain({"query": query_with_memory})
        answer: str = result["result"]
    except Exception as e:
        answer = (
            "Oops, something went wrong while contacting the RAG chain. "
            f"Technical details: `{e}`"
        )

    # Replace typing indicator with final answer
    placeholder.markdown(answer)

    # 3) Save assistant answer in history
    messages.append({
        "role": "assistant",
        "content": answer
    })