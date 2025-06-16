# **LangChain Intelligent Knowledge Base**

This project implements an intelligent Question-Answering (Q&A) system built with Python, LangChain, and OpenAI. It creates a unified knowledge base from diverse and siloed enterprise sources, including Confluence pages, technical documentation, and the source code itself.

The core of this system is a **router-retriever architecture** that intelligently directs user queries to the most relevant knowledge source before synthesizing a precise answer.


## **Core Architecture: The Router-Retriever Model**

Instead of combining all documents into a single, massive index, this system creates separate, specialized vector stores for each distinct knowledge area. This approach improves retrieval accuracy and scalability.

The query process works in three stages:



1. **Route:** A "router" LLM first analyzes the user's query. Its sole job is to decide which knowledge source (e.g., code_base, company_policy) is the most appropriate, based on curated descriptions.
2. **Retrieve:** Once the source is chosen, a specialized retriever pulls the most relevant document chunks from that source's dedicated vector store.
3. **Generate:** A final LLM call takes the original question and the retrieved documents to generate a coherent, human-readable answer.

graph TD \
    A[User Query] --> B{Router LLM}; \
    B -->|Decides: "Code Question"| C[Code Retriever]; \
    B -->|Decides: "Policy Question"| D[Confluence Policy Retriever]; \
    B -->|Decides: "Model Docs Question"| E[Model Docs Retriever]; \
    C --> F[Relevant Code Snippets]; \
    D --> G[Relevant Policy Chunks]; \
    E --> H[Relevant Model Doc Chunks]; \
    subgraph Final Answer Synthesis \
        F --> I{Answer Generation LLM}; \
        G --> I; \
        H --> I; \
        A --> I; \
    end \
    I --> J[💬 Final Answer]; \



## **How It Works**

The system is implemented in two main phases: offline indexing and online querying.


### **Phase 1: Offline Data Ingestion and Indexing**

This foundational step processes your knowledge sources and builds the vector stores. This only needs to be run initially and whenever the source documents are updated.



* **Load:** LangChain loaders (ConfluenceLoader, DirectoryLoader, etc.) ingest content from their respective sources.
* **Split:** The documents are broken down into smaller, semantically meaningful chunks. For source code, a language-aware splitter (RecursiveCharacterTextSplitter.from_language) is used to preserve the structure of functions and classes.
* **Embed & Store:** The chunks are converted into vector embeddings using OpenAI's models and stored in a local **ChromaDB** vector store. A separate, persistent store is created for each knowledge source (e.g., ./chroma_db_code, ./chroma_db_policy).


### **Phase 2: Online Querying with a Router Chain**

This is the live application that answers user questions.



* **MultiRetrievalQAChain:** LangChain's MultiRetrievalQAChain orchestrates the entire process.
* **Retriever Descriptions:** The chain is initialized with a list of all available retrievers. Each retriever is accompanied by a **critical description** that tells the router LLM what kind of questions it's good for. This acts as the "table of contents" that guides the routing decision.
* **Execution:** When a query is received, the chain first passes it to the router to select the best retriever(s). It then uses the selected retrievers to fetch context and passes everything to the final LLM to generate the answer.


## **Setup and Usage**

Follow these steps to set up and run the knowledge base system.


### **1. Prerequisites**



* Python 3.8+
* An OpenAI API Key


### **2. Installation**

Clone the repository and install the required Python packages:

git clone &lt;your-repo-url> \
cd &lt;your-repo-directory> \
pip install -r requirements.txt \


*(You will need to create a requirements.txt file containing langchain, langchain-openai, chromadb, unstructured[md], beautifulsoup4, tiktoken)*


### **3. Configuration**

Set your OpenAI API key as an environment variable.

**On macOS/Linux:**

export OPENAI_API_KEY="sk-..." \


**On Windows:**

$env:OPENAI_API_KEY="sk-..." \



### **4. Running the System**



1. Run the Indexing Script: \
First, populate the vector stores with your knowledge sources. \
python build_knowledge_base.py \
 \
This will create several chroma_db_* directories containing the indexed data.
2. Query the Knowledge Base: \
Once indexing is complete, you can start asking questions. \
python query_knowledge_base.py \



## **Customization**

This system is designed to be extensible. To add a new knowledge source:



1. **Create a Loader:** Implement the logic to load your new source documents.
2. **Update the Indexing Script:** Add a new section to build_knowledge_base.py to process and index the new source into its own dedicated ChromaDB store.
3. **Update the Router:** Add a new retriever_info entry in query_knowledge_base.py with a unique name, a clear description for the router, and the new retriever object.