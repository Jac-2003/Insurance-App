import os
import streamlit as st
from dotenv import load_dotenv
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms.ollama import Ollama
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain.embeddings.base import Embeddings

load_dotenv()

st.set_page_config(page_title="Insurance Policy Chatbot", layout="wide")

class TfidfEmbeddings(Embeddings):
    def __init__(self, max_features=5000):
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self.fitted = False
        
    def fit(self, texts):
        self.vectorizer.fit(texts)
        self.fitted = True
        
    def embed_documents(self, texts):
        if not self.fitted:
            self.fit(texts)
        embeddings = self.vectorizer.transform(texts).toarray()
        return embeddings.tolist()
    
    def embed_query(self, text):
        if not self.fitted:
            return np.zeros(self.vectorizer.max_features).tolist()
        embedding = self.vectorizer.transform([text]).toarray()[0]
        return embedding.tolist()

st.markdown("""
<style>
    .reportview-container {
        background-color: #f0f2f6;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: row;
        align-items: flex-start;
    }
    .chat-message.user {
        background-color: #e6f7ff;
        color: #000000; /* Ensure text is black */
    }
    .chat-message.bot {
        background-color: #f0f0f0;
        color: #000000; /* Ensure text is black */
    }
    .chat-message .avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        object-fit: cover;
        margin-right: 1rem;
    }
    .chat-message .message {
        flex-grow: 1;
        color: #000000; /* Ensure message text is black */
    }
    /* Apply dark text color to all text elements */
    body, p, h1, h2, h3, h4, h5, h6, li, .stTextInput>div>div>input {
        color: #000000 !important;
    }
    .stMarkdown, .stText {
        color: #000000 !important;
    }
</style>
""", unsafe_allow_html=True)

if 'conversation' not in st.session_state:
    st.session_state.conversation = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'embeddings_model' not in st.session_state:
    st.session_state.embeddings_model = None
if 'needs_rerun' not in st.session_state:
    st.session_state.needs_rerun = False

def initialize_conversation(documents):
    """Initialize the conversation chain with the provided documents."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, 
        chunk_overlap=100,  
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    
    texts = [doc.page_content for doc in chunks]
    
    embeddings = TfidfEmbeddings(max_features=1000)
    embeddings.fit(texts)  
    st.session_state.embeddings_model = embeddings
    
    vector_store = FAISS.from_documents(chunks, embeddings)
   
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer" 
    )
    
    ollama_model = st.session_state.get('ollama_model', 'llama2')
    
    llm = Ollama(
        model=ollama_model,
        temperature=0.7,
        num_ctx=2048,  
        num_predict=512,  
        base_url="http://localhost:11434"
    )
    
    conversation = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 4}),  
        memory=memory,
        return_source_documents=True,
        verbose=True,
        output_key="answer" 
    )
    
    return conversation

def process_user_message(user_question):
    st.write("Processing question...")  
    with st.spinner("Thinking... (This may take a moment with local LLM)"):
        try:
            response = st.session_state.conversation({"question": user_question})
            st.write("Got response")
        except Exception as e:
            st.error(f"Error in conversation: {str(e)}")
            return None
    
    st.session_state.chat_history.append({"role": "user", "content": user_question})
    st.session_state.chat_history.append({"role": "assistant", "content": response["answer"]})
    
    st.rerun()
    
    return response

def process_documents(uploaded_files):
    """Process uploaded PDF documents."""
    all_docs = []
    
    with st.spinner("Processing your insurance documents..."):
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(uploaded_file.read())
                temp_path = temp_file.name
            

            loader = PyPDFLoader(temp_path)
            docs = loader.load()
            
            for doc in docs:
                doc.metadata["source"] = uploaded_file.name
            
            all_docs.extend(docs)

            os.unlink(temp_path)
    
    return all_docs

def display_chat():
    """Display the chat interface with history."""
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user">
                <img class="avatar" src="https://avatars.githubusercontent.com/u/116452317?s=96&v=4">
                <div class="message">{message["content"]}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message bot">
                <img class="avatar" src="https://avatars.githubusercontent.com/u/116452326?s=96&v=4">
                <div class="message">{message["content"]}</div>
            </div>
            """, unsafe_allow_html=True)

def main():
    """Main function to render the Streamlit app."""
    if st.session_state.needs_rerun:
        st.session_state.needs_rerun = False
        st.rerun()
        
    st.title("Insurance Policy Information Chatbot")
    
    with st.sidebar:
        st.header("Configuration")
        
        st.subheader("Select Ollama Model")
        
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                available_models = [model["name"] for model in response.json()["models"]]
                if not available_models:
                    available_models = ["llama2", "mistral", "gemma:2b", "llama3"]
            else:
                available_models = ["llama2", "mistral", "gemma:2b", "llama3"]
        except:
            available_models = ["llama2", "mistral", "gemma:2b", "llama3"]
        
        model_selection = st.selectbox(
            "Choose a model",
            options=available_models,
            index=0
        )
        
        if 'ollama_model' not in st.session_state or st.session_state.ollama_model != model_selection:
            st.session_state.ollama_model = model_selection
            if st.session_state.conversation:
                st.session_state.conversation = None
                st.warning("Model changed. Please reprocess your documents.")
        
        st.subheader("Upload Insurance Documents")
        st.markdown("Upload PDF files containing insurance policy information")
        
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=["pdf"],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            if st.button("Process Documents"):
                try:
                    docs = process_documents(uploaded_files)
                    st.session_state.conversation = initialize_conversation(docs)
                    st.session_state.processing_complete = True
                    st.success(f"Successfully processed {len(docs)} pages from {len(uploaded_files)} documents")
                    
                    st.header("Sample Questions")
                    sample_questions = [
                        "What types of health insurance plans are available?",
                        "How do I file a claim for my auto insurance?",
                        "What does my home insurance policy cover?",
                        "What are the premium payment options?",
                        "What is the waiting period for my life insurance policy?"
                    ]
                    for q in sample_questions:
                        st.markdown(f"- {q}")
                        
                    st.session_state.needs_rerun = True
                except Exception as e:
                    st.error(f"Error processing documents: {str(e)}")
        
        st.subheader("Ollama Server Status")
        try:
            status = requests.get("http://localhost:11434/api/version")
            if status.status_code == 200:
                st.success(f"Connected to Ollama v{status.json().get('version', 'unknown')}")
            else:
                st.error("Ollama server is not responding properly")
        except:
            st.error("Cannot connect to Ollama server at http://localhost:11434")
            st.info("Make sure Ollama is installed and running. Visit https://ollama.ai for installation instructions.")
    
    if not st.session_state.processing_complete:
        st.info("👈 Please upload insurance policy documents in PDF format to get started.")
        
        st.header("Example Conversation")
        st.markdown("""
        <div class="chat-message user">
            <img class="avatar" src="https://avatars.githubusercontent.com/u/116452317?s=96&v=4">
            <div class="message">What types of auto insurance coverage do you offer?</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="chat-message bot">
            <img class="avatar" src="https://avatars.githubusercontent.com/u/116452326?s=96&v=4">
            <div class="message">We offer several types of auto insurance coverage:
            <br><br>
            1. <b>Liability Coverage</b>: Covers bodily injury and property damage you cause to others.
            <br>
            2. <b>Collision Coverage</b>: Pays for damage to your vehicle from accidents.
            <br>
            3. <b>Comprehensive Coverage</b>: Covers non-collision incidents like theft, weather damage, etc.
            <br>
            4. <b>Personal Injury Protection</b>: Covers medical expenses for you and your passengers.
            <br>
            5. <b>Uninsured/Underinsured Motorist Coverage</b>: Protects you from drivers with insufficient insurance.
            <br><br>
            Would you like more specific information about any of these coverage types?</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.header("Features")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### ✅ Policy Information")
            st.markdown("Get detailed information about different insurance policies")
            
            st.markdown("### ✅ Coverage Options")
            st.markdown("Learn about various coverage options and their benefits")
        
        with col2:
            st.markdown("### ✅ Claims Process")
            st.markdown("Understand how to file and track insurance claims")
            
            st.markdown("### ✅ Premium Information")
            st.markdown("Get details about premium calculations and payment options")
        
        st.header("Ollama Setup Instructions")
        st.markdown("""
        ### Before using this app:
        1. Install Ollama from [ollama.ai](https://ollama.ai)
        2. Run Ollama server on your machine
        3. Pull your preferred model using the command:
           ```
           ollama pull llama2
           ```
           (Replace 'llama2' with any model you prefer)
        
        ### Performance Tips:
        - Use smaller models like llama2:7b for faster responses
        - Models like mistral:7b often provide a good balance between speed and quality
        - Make sure your computer has adequate RAM (8GB minimum, 16GB recommended)
        - A GPU will dramatically improve performance
        
        The app will automatically connect to your local Ollama instance and use the selected model.
        """)
    
    else:
        display_chat()
 
        user_question = st.text_input("Ask a question about insurance policies:", key="user_input")
        
        if user_question:
            if st.session_state.conversation:
                try:
                    response = process_user_message(user_question)
                    if response:
                        st.session_state.user_input = ""
                except Exception as e:
                    st.error(f"Error processing your question: {str(e)}")
            else:
                st.error("Please upload and process documents first.")
        if st.button("Clear Chat"):
            st.session_state.chat_history = []
            st.session_state.needs_rerun = True

if __name__ == "__main__":
    main()
