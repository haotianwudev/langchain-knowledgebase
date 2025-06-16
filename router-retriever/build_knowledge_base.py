import os
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain.chains import MultiRetrievalQAChain
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.schema import Document
import shutil

# --- Configuration ---
# Set your OpenAI API key here
# os.environ["OPENAI_API_KEY"] = "sk-..."

# --- Helper Function to Clean Up ---
def cleanup_directories():
    """Removes the ChromaDB directories for a fresh start."""
    for dir_name in ["chroma_db_code", "chroma_db_docs", "chroma_db_policy"]:
        if os.path.exists(dir_name):
            print(f"Removing old directory: {dir_name}")
            shutil.rmtree(dir_name)

# --- 1. MOCK DATA SETUP ---
# In a real scenario, you would use LangChain's loaders for Confluence, Git, etc.
# Here, we create mock files to simulate the different knowledge sources.
print("--- Setting up mock data sources ---")
os.makedirs("knowledge_sources/code_base/utils", exist_ok=True)
os.makedirs("knowledge_sources/model_docs", exist_ok=True)
os.makedirs("knowledge_sources/confluence_policy", exist_ok=True)

# Mock Code File
with open("knowledge_sources/code_base/utils/authentication.py", "w") as f:
    f.write("""
# utils/authentication.py

class AuthHandler:
    def __init__(self, api_key):
        self.api_key = api_key

    def validate_user(self, user_id, token):
        '''
        Validates a user token against our internal system.
        Returns True for a valid user, False otherwise.
        This uses a simple check for demonstration purposes.
        '''
        if not user_id or not token:
            return False
        # In a real system, this would involve a database lookup
        return len(token) > 10

def get_admin_user():
    '''Returns the default admin user ID.'''
    return "admin_user_001"
""")

# Mock Model Docs File
with open("knowledge_sources/model_docs/model_v3_specs.md", "w") as f:
    f.write("""
# Model v3.1 Specification

## Overview
This model is a sentiment classifier trained on customer reviews.
It predicts one of three labels: POSITIVE, NEGATIVE, NEUTRAL.

## Performance
- Accuracy: 92%
- F1-Score (weighted): 0.91

## Known Limitations
The model may struggle with sarcastic or nuanced text where sentiment is ambiguous.
It is primarily trained on English language text.
""")

# Mock Confluence Policy File
with open("knowledge_sources/confluence_policy/data_handling.txt", "w") as f:
    f.write("""
# Data Handling and PII Policy

## Storing User Data
All Personally Identifiable Information (PII) must be encrypted at rest.
PII includes names, email addresses, and phone numbers.
Access to PII is restricted to personnel with a security clearance level of 3 or higher.

## Data Retention
User data should be retained for a maximum of 5 years.
After this period, it must be anonymized or securely deleted.
""")

print("--- Mock data created successfully ---\n")


# --- 2. OFFLINE INDEXING ---
# Create a separate vector store for each knowledge source.
def create_knowledge_base():
    """
    Loads documents from sources, splits them, and creates persistent ChromaDB vector stores.
    """
    print("--- Starting offline indexing process ---")

    # Initialize embeddings model
    embeddings = OpenAIEmbeddings()

    # 2a. Index the Code Base
    print("Indexing code base...")
    # Using a generic loader with a language parser is effective for code
    from langchain_community.document_loaders.generic import GenericLoader
    from langchain_community.document_loaders.parsers import LanguageParser
    
    code_loader = GenericLoader.from_filesystem(
        "./knowledge_sources/code_base",
        glob="**/*",
        suffixes=[".py"],
        parser=LanguageParser(language=Language.PYTHON, parser_threshold=10), # Lower threshold for small files
    )
    code_documents = code_loader.load()
    
    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=1000, chunk_overlap=200
    )
    code_texts = python_splitter.split_documents(code_documents)
    
    Chroma.from_documents(
        code_texts, embeddings, persist_directory="./chroma_db_code"
    )
    print("Code base indexed.")

    # 2b. Index the Model Docs
    print("Indexing model docs...")
    docs_loader = TextLoader("./knowledge_sources/model_docs/model_v3_specs.md")
    docs_documents = docs_loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs_texts = text_splitter.split_documents(docs_documents)
    
    Chroma.from_documents(
        docs_texts, embeddings, persist_directory="./chroma_db_docs"
    )
    print("Model docs indexed.")

    # 2c. Index the Confluence Policy Docs
    print("Indexing Confluence policy...")
    policy_loader = TextLoader("./knowledge_sources/confluence_policy/data_handling.txt")
    policy_documents = policy_loader.load()
    
    policy_texts = text_splitter.split_documents(policy_documents)
    
    Chroma.from_documents(
        policy_texts, embeddings, persist_directory="./chroma_db_policy"
    )
    print("Confluence policy indexed.")
    print("--- Offline indexing complete ---\n")


# --- 3. ONLINE QUERYING WITH ROUTER CHAIN ---
def query_knowledge_base():
    """
    Sets up the MultiRetrievalQAChain and runs queries against the knowledge base.
    """
    if not os.path.exists("./chroma_db_code"):
        print("Knowledge base not found. Please run the indexing first.")
        return
        
    print("--- Setting up the online querying chain ---")
    
    # Initialize the main LLM and embeddings
    llm = OpenAI(temperature=0)
    embeddings = OpenAIEmbeddings()

    # Load the persisted vector stores as retrievers
    retriever_code = Chroma(persist_directory="./chroma_db_code", embedding_function=embeddings).as_retriever()
    retriever_docs = Chroma(persist_directory="./chroma_db_docs", embedding_function=embeddings).as_retriever()
    retriever_policy = Chroma(persist_directory="./chroma_db_policy", embedding_function=embeddings).as_retriever()

    # This is the core of the router. The descriptions tell the LLM what each retriever is for.
    # This is your "table of contents".
    retriever_infos = [
        {
            "name": "code_base",
            "description": "Good for answering technical questions about function implementations, classes, and code logic.",
            "retriever": retriever_code,
        },
        {
            "name": "model_docs",
            "description": "Good for answering questions about the machine learning model's performance, specifications, and limitations.",
            "retriever": retriever_docs,
        },
        {
            "name": "company_policy",
            "description": "Good for answering questions about company policies, such as data handling, PII, and data retention rules.",
            "retriever": retriever_policy,
        },
    ]

    # Create the MultiRetrievalQAChain
    chain = MultiRetrievalQAChain.from_retrievers(
        llm,
        retriever_infos,
        verbose=True  # Set to True to see the router's decision-making process
    )

    print("--- Knowledge base is ready. Ask a question. ---")
    
    # Example queries
    # The chain will automatically route the query to the correct retriever based on the descriptions.
    
    print("\n[Query 1] What does the `validate_user` function do?")
    response = chain.invoke("What does the `validate_user` function do?")
    print(f"\n[Answer] {response['result']}")

    print("\n-----------------------------------\n")

    print("[Query 2] What is our policy on PII?")
    response = chain.invoke("What is our policy on PII?")
    print(f"\n[Answer] {response['result']}")
    
    print("\n-----------------------------------\n")

    print("[Query 3] How accurate is the latest model?")
    response = chain.invoke("How accurate is the latest model?")
    print(f"\n[Answer] {response['result']}")


if __name__ == "__main__":
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OpenAI API key not found.")
        print("Please set the OPENAI_API_KEY environment variable.")
    else:
        # Run the full process
        cleanup_directories()
        create_knowledge_base()
        query_knowledge_base()

