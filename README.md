# 🛡️ AI-Powered Insurance Policy Information Chatbot

This project is an AI-powered chatbot designed to assist users with insurance policy-related queries. It leverages Natural Language Processing (NLP) and Large Language Models (LLMs) to provide accurate, real-time responses using information extracted from insurance documents (PDFs).

---

## 🚀 Features

- ✅ Natural language understanding using LLMs
- ✅ Knowledge base generated from uploaded PDF insurance documents
- ✅ Instant answers to queries about coverage, premiums, claims, etc.
- ✅ Fallback mechanism for complex or unsupported queries
- ✅ Easy-to-use interface via Streamlit

---

## 🧠 How It Works

1. **PDF Upload**: The user uploads insurance policy PDFs.
2. **Text Extraction**: Text is extracted and split into manageable chunks.
3. **Vectorization**: Text is converted into embeddings and stored in a FAISS vector database.
4. **Query Handling**: The chatbot uses semantic search + LLM to fetch and generate accurate responses.

---

## 🛠️ Tech Stack

- **Python**
- **Streamlit** – For UI
- **OpenAI / LangChain** – For LLM and QA chain
- **FAISS** – Vector database
- **PyPDF2** – PDF reading
- **HuggingFace Sentence Transformers** – Text embedding

---

## 📦 Installation

1. **Clone the repository**
```bash
git clone https://github.com/Jac-2003/Insurance-App.git
cd Insurance-App
