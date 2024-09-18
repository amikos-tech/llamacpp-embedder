import os

import pytest
from llama_embedder import PoolingType, NormalizationType, Embedder
from huggingface_hub import hf_hub_download

DEFAULT_REPO = "leliuga/all-MiniLM-L6-v2-GGUF"
DEFAULT_TEST_MODEL = "all-MiniLM-L6-v2.Q4_0.gguf"


@pytest.fixture(scope="module")
def get_model() -> str:
    hf_hub_download(repo_id=DEFAULT_REPO, filename=DEFAULT_TEST_MODEL, local_dir="./local_model")
    return os.path.join("local_model", DEFAULT_TEST_MODEL)


def test_with_local_model_and_defaults(get_model) -> None:
    embedder = Embedder(get_model)
    embeddings = embedder.embed_texts(["Hello, world!", "Another sentence"])
    assert embeddings is not None
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 384


# TODO add property texts for normalization
def test_local_model_with_normalization(get_model) -> None:
    embedder = Embedder(get_model, normalization_type=NormalizationType.NONE)
    embeddings = embedder.embed_texts(["Hello, world!", "Another sentence"])
    assert embeddings is not None
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 384


def test_local_model_with_pooling(get_model) -> None:
    embedder = Embedder(get_model, pooling_type=PoolingType.LAST)
    embeddings = embedder.embed_texts(["Hello, world!", "Another sentence"])
    assert embeddings is not None
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 384


def test_with_hf_model_download() -> None:
    hf_repo = "ChristianAzinn/snowflake-arctic-embed-s-gguf"
    gguf_file = "snowflake-arctic-embed-s-f16.GGUF"
    embedder = Embedder(gguf_file, hf_repository=hf_repo)
    embeddings = embedder.embed_texts(["Hello, world!", "Another sentence"])
    assert embeddings is not None
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 384
