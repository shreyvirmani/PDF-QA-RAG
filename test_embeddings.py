from src.embeddings import get_embedding_model

embedding_model = get_embedding_model()

vector = embedding_model.embed_query("Machine Learning is amazing.")

print(f"Vector Length: {len(vector)}")
print(vector[:10])  # Print first 10 values