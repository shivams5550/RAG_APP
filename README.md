# 📚 PragnaAI – RAG-based AI Study Assistant

PragnaAI is an AI-powered chatbot designed to help students and professionals learn from **unstructured documents** like PDFs, textbooks, and articles. It uses **Retrieval-Augmented Generation (RAG)** to answer questions, generate quizzes, and provide citations — all grounded in your uploaded study material.

---

## 🚀 Features

- **📂 Document Upload** – Upload multiple PDFs, text files, or articles.
- **🔍 RAG-based Q&A** – Ask questions and get answers strictly from your documents.
- **📝 Quiz** – Generate a set of quiz questions to test your knowledge.
- **📑 Citations Mode** – Get answers with exact document references.
- **🧹 Clear Chat** – One-click button to reset the conversation.
---

## 🛠️ Tech Stack

- **Frontend:** [Streamlit](https://streamlit.io/)  
- **LLM:** Groq API  
- **Vector Database:** ChromaDB  
- **Embeddings:** HuggingFace Embeddings  
- **Document Processing:** LangChain Document Loaders & Text Splitters  

---

## 📦 Installation

### 1️⃣ Install Python (if not installed)

1. Download the latest **Python 3.10+** from [python.org/downloads](https://www.python.org/downloads/).  
2. During installation, make sure to **tick**:
   - ✅ *"Add Python to PATH"*  
   - ✅ *"Install pip"*
3. Verify installation:
```bash
python --version
```

### 2️⃣ Clone the Repository
```bash
git clone https://github.com/shivams5550/RAG_APP.git
cd RAG_APP
```

### 3️⃣ Create and Activate a Virtual Environment
```cmd
python -m venv venv
venv\Scripts\activate
```

### 4️⃣ Install Dependencies
```cmd
pip install -r requirement.txt
```

### 5️⃣ Create .env File
```cmd
GROQ_API_KEY=your_groq_api_key_here
```

### 6️⃣ Run the App
```cmd
streamlit run RAG.py
```


### 💡 Usage

Upload Documents – Click "Upload" to add your PDFs or text files.

Select Mode:

Q&A – Ask questions about your document.

Quiz – Generate a set of quiz questions.

Citation – Get answers with document sources.

Chat – Type your query in the chat input box.

Clear Chat – Use the "Clear Chat" button to reset.










