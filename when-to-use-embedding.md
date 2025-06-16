# When to Use Embeddings vs. Direct Text Input for LLMs

## Decision Framework

This framework outlines when to use direct LLM input versus embeddings, considering data size and technical factors.

| Data Size       | Approach               | Typical Scenarios               | Technical Considerations         |
|-----------------|------------------------|---------------------------------|----------------------------------|
| **<1K tokens**  | Direct LLM input       | Short Q&A, code snippets       | Context window limits            |
| **1K-10K tokens**| Hybrid embedding       | Medium documents               | Precision/cost balance           |
| **>10K tokens** | Must use embeddings    | Knowledge bases, long docs     | Avoid information overload       |

## Key Limitations

### 1. LLM Context Window Constraints

LLMs have limitations on the amount of text they can process at once. These context window sizes vary by model:

- **GPT-4**: 128K tokens (practical limit ~100K)
- **Llama 3**: 8K-32K tokens
- **Claude 3**: 200K tokens

### 2. Performance Degradation

As input length increases, accuracy tends to decrease. This is an empirical observation and varies by model.

**Accuracy vs. input length (empirical data):**

| Length (tokens) | Accuracy (%) |
|---|---|
| 1k | 92 |
| 4k | 85 |
| 8k | 76 |
| 32k | 58 |
| 128k | 41 |

### 3. Cost Comparison ($/1K tokens)

The cost of processing text differs significantly between direct input and embedding-based retrieval.

| Method                   | 1K tokens | 10K tokens |
|--------------------------|-----------|-------------|
| Direct GPT-4            | $0.03     | $0.30      |
| Embedding+Retrieval     | $0.0006 + $0.003 | $0.006 + $0.03 |

## Implementation Guide

### Scenario 1: Short Text (<1K tokens)

For short texts, direct input to the LLM is suitable.

```python
response = llm.invoke(f"""
Answer directly without retrieval:
Question: {question}
Context: {short_text}  # <800 tokens
""")
```

### Scenario 2: Medium Text (1K-10K tokens)

A hybrid approach, extracting key sections, can be effective.

```python
# Hybrid approach
key_paragraphs = [p for p in split_text(text) if is_relevant(p, question)]
response = llm.invoke(f"""
Question: {question}
Key paragraphs: {key_paragraphs}  # <2K tokens
Full document (reference): {text}  
""")
```

### Scenario 3: Long Text (>10K tokens)

Embedding-based retrieval is mandatory for long texts.

```python
# Mandatory embedding retrieval
retriever = vector_db.as_retriever(search_kwargs={"k": 5})
docs = retriever.get_relevant_documents(question)
response = llm.invoke(f"Answer based on:\n{docs}\nQuestion: {question}")
```

## Optimization Techniques

### 1. Dynamic Chunking

Semantic chunking can improve text splitting.

```python
from langchain_text_splitters import SemanticChunker

splitter = SemanticChunker(
    embeddings=HuggingFaceEmbeddings(),
    breakpoint_threshold_type="percentile",
    percentile_threshold=95
)
```

### 2. Hierarchical Attention

Extracting a table of contents (TOC) and then selecting relevant sections can enhance retrieval.

```python
# First extract TOC, then select sections
toc = llm.invoke(f"Extract TOC from:\n{text[:2000]}")
selected_sections = llm.invoke(f"Select relevant sections for: {question}\nTOC: {toc}")
```

### 3. Compression Retrieval

Contextual compression retrieval can reduce the amount of text retrieved.

```python
from langchain.retrievers import ContextualCompressionRetriever

compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vector_db.as_retriever()
)
```

## Rule of Thumb

Here are some practical guidelines for different document types:

- **Legal contracts:** Always embed (50K+ tokens avg)
- **Technical docs:** Embed if >5 pages
- **Chat logs:** Direct input (unless cross-session)
- **Research papers:** Section-level embedding