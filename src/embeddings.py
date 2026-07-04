import torch
from langchain_huggingface import HuggingFaceEmbeddings


def get_embedding_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": device},
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 64,
        },
    )

    return embedding_model
