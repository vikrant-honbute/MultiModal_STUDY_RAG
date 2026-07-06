"""
Small script to test local embeddings without HF API.
Run with:
    python -m src.tools.test_embeddings
"""
from src.vectorstore.embeddings import EmbeddingModel


def main():
    texts = [
        "This is a short test sentence.",
        "Another example sentence for embeddings.",
    ]

    em = EmbeddingModel(use_local=True)
    vecs = em.embed(texts)
    print(f"Computed {len(vecs)} vectors, dim={len(vecs[0]) if vecs else 0}")


if __name__ == "__main__":
    main()
