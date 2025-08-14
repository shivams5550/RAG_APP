
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
from langchain.schema.runnable import RunnableBranch
from langchain_groq import ChatGroq
from sentence_transformers import CrossEncoder



load_dotenv()



langsmith_tracing = os.getenv('LANGSMITH_TRACING')
groq_api_key = os.getenv('GROQ_API_KEY')
langsmith_api_key = os.getenv('LANGSMITH_API_KEY')
langsmith_project = os.getenv('LANGSMITH_PROJECT')



client = Groq(api_key = groq_api_key)



model_name = "openai/gpt-oss-120b"




embedding_model_name = "thenlper/gte-large"




prompt = ChatPromptTemplate.from_messages([
    ("system", """ You are a specialized AI study assistant.
You will receive input that contains the necessary context for answering a user’s question.
The context will begin with the token: ###Context.
This context contains relevant excerpts from documents such as PDFs, textbooks, or articles.

User questions will begin with the token: ###Question.

Your task:
- Answer the user’s question using only the information provided in the context.
- Do not mention the context or the tokens in your answer.
- Provide clear and educationally helpful responses.
- If possible, explain concepts in a way that aids understanding and retention.
- If the answer is not present in the context, respond with: "I don't know".

Your goal is to help the user learn and understand the material while staying strictly grounded in the provided context.
"""
),
    ("user", """###Context
    {context}

    ###Question
    {question}""")
])



citation_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a specialized AI study assistant.
Use the provided context to answer the question AND provide the sources at the end under 'Sources:'.
Only use the provided sources; do not fabricate references."""),
    ("user", """###Context
{context}

###Question
{question}

###Sources
{sources}""")
])



output_parser = StrOutputParser() #For parcing the output of LLM as simple string





# Function for processing the uploaded documents (loading, chunking, embedding, vectorDB creation)

def process_uploaded_files(uploaded_files):

        documents = []
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 512, chunk_overlap = 16)

        #loop to save the files temporarily, putting the condition of pdf and txt to choose the suitable loader, splitting the doc using the recursive technique then add those chunks in the global documents variable.
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False) as temp:
                temp.write(uploaded_file.read())
                temp_path = temp.name

            if uploaded_file.name.endswith(".pdf"):
                loader = PyPDFLoader(temp_path)

            else:
                loader = TextLoader(temp_path)

            loaded_documents = loader.load()
            split_documents = text_splitter.split_documents(loaded_documents)
            documents.extend(split_documents) #loaded the chunks to global docs 

        #embedding
        embedding_model = SentenceTransformerEmbeddings(model_name = embedding_model_name)

        #Creating vectorDB of the chunks using ChromaDB
    
        #initialising the empty vectorDB
        st.session_state.vectorstore = Chroma(collection_name = "user_documents",
                             embedding_function = embedding_model)
        
        st.session_state.vectorstore.add_documents(documents)
        st.session_state.retriever = st.session_state.vectorstore.as_retriever(search_type = "similarity", search_kwargs = {"k":20})

        return len(documents)




if "reranker" not in st.session_state:
    st.session_state.reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")
    
def retrieve_and_rerank(question, top_k_initial, top_k_final):
    if "retriever" not in st.session_state:
        st.warning("Please upload your documents first.")
        return []
    
    #Retrieving the initial top_k sentences from the vectorDB based on the similarity score.
    retrieved_documents = st.session_state.retriever.get_relevant_documents(question, k=top_k_initial)
    
    pairs = [(question, doc.page_content) for doc in retrieved_documents]
    
    #Scoring the pairs using reranker
    scores = st.session_state.reranker.predict(pairs)
    
    reranked_documents = [
    doc for _, doc in sorted(zip(scores, retrieved_documents), key=lambda x: x[0], reverse=True)
    ][:top_k_final]

    
    return reranked_documents





#need this for using groq with langchain

llm = ChatGroq(model = model_name, api_key = os.environ["GROQ_API_KEY"], temperature = 0)




#function for making the prompt that needs to be send to llm (context + question)

def retrieve_context(inputs):
    question = inputs["question"]
    relevant_chunks = retrieve_and_rerank(
        question,
        top_k_initial= st.session_state.top_k_initial,
        top_k_final= st.session_state.top_k_final                                  
                                        )
    context = ". ".join([d.page_content for d in relevant_chunks])
    sources = [f"Source: {getattr(d.metadata, 'source', 'Unknown')}" for d in relevant_chunks]
    return {"context": context, "question": question, "sources": "\n".join(sources)}
    



#building the chain to integrate the whole pipeline process


qa_chain = RunnableLambda(retrieve_context) | prompt | llm | StrOutputParser()
citation_chain = RunnableLambda(retrieve_context) | citation_prompt | llm | StrOutputParser()
#Wrapping the retrieve_context function in RunnableLambda beacuse we can't add manually made functions directly to the langchain's chain. we need RunnableLambda to wrap the function to work as an in-build langchain function





#MAKING THE CHATBOT's UI



st.set_page_config(
    page_title="PragnaAI",
    page_icon="📚",
    layout="wide",
)

# Custom CSS
st.markdown("""
<style>
    html, body, [class*="stApp"] {
        background-color: #001621 !important;
        font-family: 'Segoe UI', sans-serif;
        color: white;
    }
    
    input[type=range] {
     accent-color: #06d0d4;
    }
    
    section[data-testid="stSidebar"] {
        background-color: #01283d !important;
        color: white !important;
    }
    
 
    .sidebar .sidebar-content {
        background-color: #01283d !important;
        color: white !important;
    }
    .big-font {
        font-size: 50px !important;
        font-weight: bold;
        color: #06d0d4;
        text-align: center;
    }
    
    .subheader-font {
        font-size: 28px !important;
        font-weight: bold;
        color: #06d0d4;
    }
    
    .sub-text {
        font-size: 16px;
        color: white;
        text-align: center;
        margin:7%;
    }
    .stButton>button {
        background-color: #06d0d4;
        color: white;
        border-radius: 8px;
        font-weight: bold;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #04a5a8;
    }
    .stSlider label {
        color: white !important;
    }
    
    
    .chat-container {
        height: 400px; /* Fixed height */
        overflow-y: auto; /* Enable vertical scroll */
        padding: 10px;
        background-color: #028D93;
        border: 2px solid #06d0d4;
        border-radius: 10px;
    }
    
    .user-msg {
        background-color: #06d0d4;
        color: white;
        padding: 10px 15px;
        border-radius: 15px;
        max-width: 100%;
        margin-left: auto;
        margin-bottom: 10px;
    }
    
    .ai-msg {
        background-color: #01283d;
        color: white;
        padding: 10px 15px;
        border-radius: 15px;
        max-width: 100%;
        margin-right: auto;
        margin-bottom: 10px;
    }
    
    .chat-input {
    position: sticky;
    bottom: 0;
    background-color: #01283d;
    padding: 10px;
    }   
    
    
</style>
""", unsafe_allow_html=True)


# Header
st.markdown('<p class="big-font">📚 PragnaAI</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">Your personal AI study companion by AIonOS — upload documents and start learning smarter today.</p>', unsafe_allow_html=True)

with st.sidebar:
        mode = st.selectbox("Select Mode", ["Q&A", "Citation"])
        

left_col, right_col = st.columns([2,3])

#left side of the page:
with left_col:
    st.markdown('<p class="subheader-font">📂 Document Panel</p>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload your PDFs or text files here! ",
        type = ["pdf","txt"],
        accept_multiple_files = True)
    
    st.markdown('<p class="subheader-font">⚙️ Retrieval Settings</p>', unsafe_allow_html=True)
    st.session_state.top_k_initial = st.slider(
        "Initial relevant top_k chunks",
        min_value = 5, max_value = 20, value = 20, step = 1
    )
    
    st.session_state.top_k_final = st.slider(
        "Final Reranked top_k chunks",
        min_value = 1, max_value = st.session_state.top_k_initial, value = 10, step = 1
    )
    
    
    if uploaded_files:
        if st.button("Process Documents"):
            num_chunks = process_uploaded_files(uploaded_files)
            st.success(f"Processed {num_chunks} chunks from uploaded files.")
        
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun() 
            
            
            
            
with right_col:
    st.markdown('<p class="subheader-font">💬 Chat with Pragna</p>', unsafe_allow_html=True)
    
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
        

    
    user_input = st.text_input("Hi! I am your Study Buddy. How may I help you? ", key = "chat_input", label_visibility="collapsed")
    

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
            st.session_state.messages.append({"role": "assistant","content": answer})

    
    # Build HTML for chat messages
    chat_html = "<div class='chat-container'>"
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            chat_html += f"<div class='user-msg'>👤 {msg['content']}</div>"
        else:
            chat_html += f"<div class='ai-msg'>🤖 {msg['content']}</div>"
    chat_html += "</div>"
    
        # Render chat container
    st.markdown(chat_html, unsafe_allow_html=True)
    
    
    





