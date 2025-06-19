import os
from openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

# --- 1. Setup and Initialization ---
# os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"

try:
    client = OpenAI()
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    print("Please make sure your OPENAI_API_KEY is set correctly.")
    exit()
    
# We still need an embedding object for the final search step,
# so we'll use the custom wrapper from Solution 1 here as well.
# Alternatively, you could handle the final search embedding manually too.
class CustomOpenAIEmbeddings:
    def __init__(self, client, model="text-embedding-3-small"):
        self.client = client
        self.model = model
    def embed_documents(self, texts):
        texts = [text.replace("\n", " ") for text in texts]
        response = self.client.embeddings.create(input=texts, model=self.model)
        return [embedding.embedding for embedding in response.data]
    def embed_query(self, text):
        response = self.client.embeddings.create(input=[text.replace("\n", " ")], model=self.model)
        return response.data[0].embedding

embedding_function_for_search = CustomOpenAIEmbeddings(client=client)

# --- 2. Manual Embedding Generation ---

# Prepare your document texts
sample_texts = [
    "The Wright brothers made the first sustained flight in 1903.",
    "The capital of France is Paris, known for the Eiffel Tower.",
    "Quantum computing leverages principles of quantum mechanics.",
    "The first successful airplane was named the Wright Flyer."
]
# We only need the string content for this step
texts_to_embed = [text.replace("\n", " ") for text in sample_texts]

# Use your client to generate embeddings in a batch
embeddings_response = client.embeddings.create(input=texts_to_embed, model="text-embedding-3-small")
raw_embeddings = [embedding.embedding for embedding in embeddings_response.data]

# Pair the original texts with their embeddings
text_embedding_pairs = list(zip(sample_texts, raw_embeddings))

# --- 3. Create Vector Store with FAISS.from_embeddings ---
# Note: We pass the embedding function needed for future queries separately.
db = FAISS.from_embeddings(text_embeddings=text_embedding_pairs, embedding=embedding_function_for_search)

print("FAISS vector store created successfully from pre-computed embeddings!")

# --- 4. Perform a search ---
query = "What is the history of aviation?"
results = db.similarity_search(query)

print("\nSearch Results:")
for doc in results:
    print(f"- {doc.page_content}")