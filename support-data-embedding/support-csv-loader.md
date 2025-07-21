# Loading CSV Support Data into FAISS with Metadata

Here's how to load your CSV file where each row represents a support item, with the description as content and other columns as metadata:

## Step 1: Install Required Packages

```bash
pip install pandas langchain faiss-cpu sentence-transformers
```

## Step 2: Load CSV and Prepare Documents

```python
import pandas as pd
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

def load_csv_to_faiss(csv_path: str, 
                     content_column: str, 
                     metadata_columns: list, 
                     save_path: str, 
                     chunk_size: int = 1000,
                     chunk_overlap: int = 200):
    """
    Load CSV data into FAISS vector store with proper metadata handling
    
    Args:
        csv_path: Path to CSV file
        content_column: Column name containing the text to embed
        metadata_columns: List of columns to include as metadata
        save_path: Where to save the FAISS index
        chunk_size: Size of text chunks (if content is long)
        chunk_overlap: Overlap between chunks
    """
    # Load CSV file
    df = pd.read_csv(csv_path)
    
    # Initialize embedding model
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Prepare documents
    documents = []
    for _, row in df.iterrows():
        # Extract metadata
        metadata = {col: row[col] for col in metadata_columns}
        
        # Handle potential NaN values in metadata
        for key in metadata:
            if pd.isna(metadata[key]):
                metadata[key] = None  # Replace NaN with None
            
        # Create document with content and metadata
        doc = Document(
            page_content=row[content_column],
            metadata=metadata
        )
        documents.append(doc)
    
    # Split documents if needed (for long descriptions)
    if chunk_size > 0:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        documents = text_splitter.split_documents(documents)
    
    # Create and save FAISS index
    db = FAISS.from_documents(documents, embedding_model)
    db.save_local(save_path)
    print(f"FAISS index created with {len(documents)} documents, saved to {save_path}")
    return db
```

## Step 3: Example Usage

```python
# Example usage
db = load_csv_to_faiss(
    csv_path="support_data.csv",
    content_column="description",  # Column with support content
    metadata_columns=["date", "label", "product", "priority"],  # Columns to keep as metadata
    save_path="support_faiss_index",
    chunk_size=1000  # Set to 0 if you don't want chunking
)
```

## Step 4: Loading and Querying the Index

```python
def load_faiss_index(folder_path: str):
    """Load existing FAISS index"""
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    return FAISS.load_local(folder_path, embedding_model)

# Load the index
db = load_faiss_index("support_faiss_index")

# Example query with metadata filtering
results = db.max_marginal_relevance_search(
    "How to reset password?",
    filter={"label": "account_management", "product": "web_app"},
    k=5
)

for doc in results:
    print(f"Content: {doc.page_content[:100]}...")
    print(f"Metadata: {doc.metadata}")
    print("---")
```

## Handling Different Data Types in Metadata

If your metadata contains non-string types (dates, numbers, etc.), you might want to convert them:

```python
def convert_metadata_types(metadata: dict) -> dict:
    """Convert metadata values to appropriate types for FAISS filtering"""
    converted = {}
    for key, value in metadata.items():
        if pd.isna(value):
            converted[key] = None
        elif isinstance(value, (int, float)):
            converted[key] = float(value)
        elif isinstance(value, str) and value.isdigit():
            converted[key] = float(value)
        else:
            converted[key] = str(value)
    return converted

# Then modify the document creation in load_csv_to_faiss:
metadata = {col: row[col] for col in metadata_columns}
metadata = convert_metadata_types(metadata)
```

## Complete Example with Date Handling

```python
from datetime import datetime

def load_csv_with_dates(csv_path, date_columns):
    """Load CSV and properly parse date columns"""
    df = pd.read_csv(csv_path)
    
    # Convert date columns
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col]).dt.strftime('%Y-%m-%d')  # Convert to string format
    
    return df

# Usage:
df = load_csv_with_dates("support_data.csv", ["created_date", "updated_date"])
```

This approach gives you:
1. Proper loading of CSV data into FAISS
2. Preservation of all metadata columns
3. Handling of different data types in metadata
4. Optional text chunking for long descriptions
5. Easy querying with metadata filters

Remember that FAISS metadata filtering works best with:
- String values (for exact matching)
- Numeric values (for range queries)
- Avoid using very high-cardinality metadata for filtering