import os
import tempfile
from dotenv import load_dotenv
import streamlit as st
from groq import Groq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_groq import ChatGroq
from sentence_transformers import CrossEncoder
import json
import re

# ==================== ENV ====================
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')
client = Groq(api_key=groq_api_key)

# ==================== MODELS ====================
model_name = "openai/gpt-oss-120b"
embedding_model_name = "thenlper/gte-large"

# ==================== PROMPTS ====================
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a specialized AI study assistant.
You will receive input that contains the necessary context for answering a user’s question.
The context will begin with the token: ###Context.
User questions will begin with the token: ###Question.
Answer ONLY from the context. If not present, respond with: "I don't know"."""), 
    ("user", """###Context
{context}

###Question
{question}""")
])

citation_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a study assistant.
Use the provided context to answer the question AND list the sources under 'Sources:'.
Only use the provided sources; do not fabricate references."""), 
    ("user", """###Context
{context}

###Question
{question}

###Sources
{sources}""")
])

quiz_prompt = ChatPromptTemplate.from_template(
    """
    You are a quiz generator. Based on the following context from documents, create {num_questions} multiple-choice questions
    on the topic: {question}.
    
    Context:
    {context}

    For each question, provide:
    - "question": the quiz question
    - "options": a list of 4 choices
    - "answer": the correct choice (must exactly match one option)

    Return the output strictly as a JSON list of questions.
    """
)

# ==================== HELPERS ====================
output_parser = StrOutputParser()
llm = ChatGroq(model=model_name, api_key=groq_api_key, temperature=0)

if "reranker" not in st.session_state:
    st.session_state.reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")

def process_uploaded_files(uploaded_files):
    documents = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=16)

    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            temp.write(uploaded_file.read())
            temp_path = temp.name

        if uploaded_file.name.endswith(".pdf"):
            loader = PyPDFLoader(temp_path)
        else:
            loader = TextLoader(temp_path)

        loaded_docs = loader.load()
        split_docs = text_splitter.split_documents(loaded_docs)
        documents.extend(split_docs)

    embedding_model = SentenceTransformerEmbeddings(model_name=embedding_model_name)

    st.session_state.vectorstore = Chroma(
        collection_name="user_documents",
        embedding_function=embedding_model
    )

    st.session_state.vectorstore.add_documents(documents)
    st.session_state.retriever = st.session_state.vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 20}
    )
    return len(documents)

def retrieve_and_rerank(question, top_k_initial, top_k_final):
    if "retriever" not in st.session_state:
        st.warning("Please upload your documents first.")
        return []

    retrieved = st.session_state.retriever.get_relevant_documents(question, k=top_k_initial)
    pairs = [(question, doc.page_content) for doc in retrieved]
    scores = st.session_state.reranker.predict(pairs)

    reranked = [
        doc for _, doc in sorted(zip(scores, retrieved), key=lambda x: x[0], reverse=True)
    ][:top_k_final]

    return reranked

def retrieve_context(inputs):
    question = inputs["question"]
    relevant_chunks = retrieve_and_rerank(
        question, top_k_initial=st.session_state.top_k_initial,
        top_k_final=st.session_state.top_k_final
    )
    context = ". ".join([d.page_content for d in relevant_chunks])
    sources = [f"Source: {getattr(d.metadata, 'source', 'Unknown')}" for d in relevant_chunks]
    return {"context": context, "question": question, "sources": "\n".join(sources)}

# Chains
qa_chain = RunnableLambda(retrieve_context) | prompt | llm | StrOutputParser()
citation_chain = RunnableLambda(retrieve_context) | citation_prompt | llm | StrOutputParser()

# ==================== UI ====================
st.set_page_config(page_title="PragnaAI", page_icon="📚", layout="wide")

# CSS
st.markdown("""
<style>
    html, body, [class*="stApp"] {
        background-color: #001621 !important;
        font-family: 'Segoe UI', sans-serif;
        color: white;
    }
    .big-font { font-size: 50px !important; font-weight: bold; color: #06d0d4; text-align: center; }
    .subheader-font { font-size: 28px !important; font-weight: bold; color: #06d0d4; }
    .sub-text { font-size: 16px; color: white; text-align: center; margin:7%; }
    .chat-container { height: 400px; overflow-y: auto; padding: 10px;
                      background-color: #028D93; border: 2px solid #06d0d4; border-radius: 10px; }
    .user-msg { background-color: #06d0d4; color: white; padding: 10px 15px;
                border-radius: 15px; margin-left: auto; margin-bottom: 10px; }
    .ai-msg { background-color: #01283d; color: white; padding: 10px 15px;
              border-radius: 15px; margin-right: auto; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="big-font">📚 PragnaAI</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">Your personal AI study companion — upload documents and start learning smarter today.</p>', unsafe_allow_html=True)

with st.sidebar:
    mode = st.selectbox("Select Mode", ["Q&A", "Citation", "Quiz"])

left_col, right_col = st.columns([2, 3])

# -------- Left (Docs + Settings) --------
with left_col:
    st.markdown('<p class="subheader-font">📂 Document Panel</p>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader("Upload PDFs or TXT files", type=["pdf", "txt"], accept_multiple_files=True)

    st.markdown('<p class="subheader-font">⚙️ Retrieval Settings</p>', unsafe_allow_html=True)
    st.session_state.top_k_initial = st.slider("Initial top_k chunks", 5, 20, 20, 1)
    st.session_state.top_k_final = st.slider("Final reranked top_k", 1, st.session_state.top_k_initial, 10, 1)

    if uploaded_files:
        if st.button("Process Documents"):
            num_chunks = process_uploaded_files(uploaded_files)
            st.success(f"Processed {num_chunks} chunks from uploaded files.")

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.quiz = []
        st.session_state.user_answers = {}
        st.rerun()

# -------- Right (Chat + Quiz) --------
with right_col:
    if mode in ["Q&A", "Citation"]:
        st.markdown('<p class="subheader-font">💬 Chat with Pragna</p>', unsafe_allow_html=True)

        if "messages" not in st.session_state:
            st.session_state.messages = []

        user_input = st.text_input("Hi! I am your Study Buddy. How may I help you?", key="chat_input", label_visibility="collapsed")

        if user_input:
            if "retriever" not in st.session_state:
                st.warning("⚠️ Please upload and process your documents first.")
            else:
                with st.spinner("🤔 Thinking..."):
                    if mode == "Q&A":
                        answer = qa_chain.invoke({"question": user_input})
                    elif mode == "Citation":
                        answer = citation_chain.invoke({"question": user_input})

                st.session_state.messages.append({"role": "user", "content": user_input})
                st.session_state.messages.append({"role": "assistant", "content": answer})

        # Display chat
        chat_html = "<div class='chat-container'>"
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                chat_html += f"<div class='user-msg'>👤 {msg['content']}</div>"
            else:
                chat_html += f"<div class='ai-msg'>🤖 {msg['content']}</div>"
        chat_html += "</div>"
        st.markdown(chat_html, unsafe_allow_html=True)

    elif mode == "Quiz":
        st.markdown('<p class="subheader-font">📝 Quiz Mode</p>', unsafe_allow_html=True)

        if "quiz" not in st.session_state:
            st.session_state.quiz = []
            st.session_state.user_answers = {}

        user_input = st.text_input("Enter a topic (you can also add number of questions, e.g., 'Physics 5'):").strip()

        if st.button("Generate Quiz"):
            if "retriever" not in st.session_state:
                st.warning("⚠️ Please upload and process your documents first.")
            elif not user_input:
                st.warning("⚠️ Please enter a topic for the quiz.")
            else:
                # Extract number if given, else default = 10
                match = re.search(r"(\d+)", user_input)
                if match:
                    num_q = int(match.group(1))
                    quiz_topic = re.sub(r"\d+", "", user_input).strip()
                else:
                    num_q = 10
                    quiz_topic = user_input

                with st.spinner("📝 Generating quiz..."):
                    context_data = retrieve_context({"question": quiz_topic})
                    quiz_input = {
                        "context": context_data["context"],
                        "num_questions": num_q,
                        "question": quiz_topic
                    }
                    raw_quiz = quiz_prompt.format(**quiz_input)
                    quiz_response = llm.invoke(raw_quiz)
                    quiz_text = quiz_response.content if hasattr(quiz_response, "content") else str(quiz_response)

                    try:
                        st.session_state.quiz = json.loads(quiz_text)
                        st.session_state.user_answers = {}
                        st.success(f"✅ Quiz Generated with {len(st.session_state.quiz)} questions!")
                    except:
                        st.error("⚠️ Failed to parse quiz JSON. Showing raw output:")
                        st.text(quiz_text)

        if st.session_state.quiz:
            st.markdown("### Answer the following questions:")
            for i, q in enumerate(st.session_state.quiz):
                st.write(f"**Q{i+1}. {q['question']}**")
                st.session_state.user_answers[i] = st.radio(
                    "Select an answer:", q["options"], key=f"q{i}"
                )

            if st.button("Submit Answers"):
                correct = 0
                results = []
                for i, q in enumerate(st.session_state.quiz):
                    user_ans = st.session_state.user_answers.get(i, None)
                    if user_ans == q["answer"]:
                        correct += 1
                        results.append(f"Q{i+1}: ✅ Correct")
                    else:
                        results.append(f"Q{i+1}: ❌ Wrong (Correct: {q['answer']})")

                st.markdown(f"### 🏆 You got {correct}/{len(st.session_state.quiz)} correct!")
                st.markdown("\n".join(results))
