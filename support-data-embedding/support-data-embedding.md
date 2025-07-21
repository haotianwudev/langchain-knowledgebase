# Saving Support Data to FAISS Vector Store with Metadata and Building a RAG Chatbot

Here's a comprehensive approach to store your support data in a FAISS vector store with metadata using LangChain, and later build a metadata-filterable RAG chatbot:

## Step 1: Prepare Your Data and Environment

First, install required packages:
```bash
pip install langchain faiss-cpu sentence-transformers
# or for GPU:
pip install langchain faiss-gpu sentence-transformers
```

## Step 2: Create and Save the FAISS Index with Metadata

```python
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from typing import List, Dict

# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",  # Good general-purpose model
    model_kwargs={'device': 'cpu'},  # or 'cuda' if available
    encode_kwargs={'normalize_embeddings': True}
)

def create_and_save_faiss_index(support_data: List[Dict], save_path: str):
    """
    Create FAISS index from support data with metadata and save to disk
    
    Args:
        support_data: List of dictionaries with 'description', 'metadata', etc.
        save_path: Where to save the FAISS index
    """
    # Create LangChain Documents with metadata
    documents = []
    for item in support_data:
        doc = Document(
            page_content=item['description'],  # The main text to embed
            metadata=item['metadata']  # All metadata you want to filter on later
        )
        documents.append(doc)
    
    # Create FAISS index
    db = FAISS.from_documents(documents, embedding_model)
    
    # Save to disk
    db.save_local(save_path)
    print(f"FAISS index saved to {save_path}")

# Example usage:
support_data = [
    {
        "description": "How to reset your password...", 
        "metadata": {
            "category": "account",
            "product": "web_app",
            "support_level": "basic",
            "doc_id": 101
        }
    },
    # ... more support items
]

create_and_save_faiss_index(support_data, "support_faiss_index")
```

## Step 3: Load the Index and Create a Metadata-Filterable Retriever

```python
def load_faiss_index(load_path: str, embedding_model):
    """Load FAISS index from disk"""
    return FAISS.load_local(load_path, embedding_model)

# Load the index
db = load_faiss_index("support_faiss_index", embedding_model)

def get_metadata_filterable_retriever(db, top_k=5):
    """Create a retriever that can filter by metadata"""
    # Create base retriever
    retriever = db.as_retriever(search_kwargs={"k": top_k})
    
    # Add metadata filtering capability
    def filtered_retrieve(query, metadata_filter=None):
        if metadata_filter:
            # Apply metadata filter
            docs = db.max_marginal_relevance_search(
                query, 
                filter=metadata_filter,
                k=top_k
            )
        else:
            # Regular search
            docs = db.max_marginal_relevance_search(query, k=top_k)
        return docs
    
    return filtered_retrieve

# Example usage:
retriever = get_metadata_filterable_retriever(db)

# Search without filter
results = retriever("How to reset password?")

# Search with metadata filter
filtered_results = retriever(
    "How to reset password?",
    metadata_filter={"category": "account", "product": "web_app"}
)
```

## Step 4: Build the RAG Chatbot with Metadata Filtering

```python
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI  # or any other LLM you prefer
from langchain.prompts import PromptTemplate

def create_rag_chatbot(db, llm):
    # Create the base QA chain
    prompt_template = """Use the following support documentation to answer the question. 
    If you don't know the answer, say you don't know. Don't make up an answer.

    Context: {context}
    Question: {question}
    Helpful Answer:"""
    
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(),
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )
    
    def ask_question(question, metadata_filter=None):
        if metadata_filter:
            # Apply metadata filter by temporarily modifying the retriever
            original_retriever = qa_chain.retriever
            qa_chain.retriever = get_metadata_filterable_retriever(db)(question, metadata_filter)
            result = qa_chain({"query": question})
            qa_chain.retriever = original_retriever
        else:
            result = qa_chain({"query": question})
        
        return {
            "answer": result["result"],
            "sources": [doc.metadata for doc in result["source_documents"]]
        }
    
    return ask_question

# Initialize LLM
llm = OpenAI(temperature=0)  # or any other LLM

# Create chatbot
chatbot = create_rag_chatbot(db, llm)

# Example usage:
response = chatbot("How do I reset my password?")
print(response["answer"])
print("Sources:", response["sources"])

# With metadata filtering
filtered_response = chatbot(
    "How do I reset my password?",
    metadata_filter={"product": "mobile_app"}
)
```

## Step 5: Deploy as a Full Chatbot (Optional)

For a more complete chatbot experience with conversation history:

```python
from langchain.memory import ConversationBufferMemory

def create_chatbot_with_memory(db, llm):
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=db.as_retriever(),
        memory=memory,
        chain_type="stuff",
        combine_docs_chain_kwargs={"prompt": PROMPT}
    )
    
    def ask_question(question, metadata_filter=None):
        if metadata_filter:
            # Apply metadata filter
            docs = db.max_marginal_relevance_search(
                question, 
                filter=metadata_filter,
                k=5
            )
            context = "\n".join([doc.page_content for doc in docs])
            result = qa_chain({"question": question, "context": context})
        else:
            result = qa_chain({"question": question})
        
        return {
            "answer": result["answer"],
            "chat_history": result["chat_history"],
            "sources": [doc.metadata for doc in docs] if metadata_filter else []
        }
    
    return ask_question
```

## Handling Large Documents Without Truncation

If your support items are very large and getting truncated:

1. **Chunk your documents** before embedding:
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)

split_docs = text_splitter.split_documents(documents)
db = FAISS.from_documents(split_docs, embedding_model)
```

2. **Use a larger embedding model** that supports longer sequences:
```python
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",  # 384 token max
    # or for longer sequences:
    # model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)
```

This implementation gives you a complete system to:
- Store support items with metadata in FAISS
- Retrieve information with metadata filtering
- Build a RAG chatbot that can filter support content by metadata
- Handle large documents through chunking
- Maintain conversation history if needed