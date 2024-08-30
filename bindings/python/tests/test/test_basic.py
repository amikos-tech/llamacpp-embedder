from llama_embedder import LlamaEmbedder, PoolingType, NormalizationType


def test_basic():
    # Using without a context manager
    embedder = LlamaEmbedder("/Users/tazarov/Downloads/all-MiniLM-L6-v2.F32.gguf")
    embeddings = embedder.embed(["Hello, world!", "Another sentence"], norm=NormalizationType.EUCLIDEAN)
    assert embeddings is not None
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 384
# TODO we need more tests
