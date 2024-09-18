import os
from typing import List

import pytest
from llama_embedder import LlamaEmbedder, PoolingType, NormalizationType
from huggingface_hub import hf_hub_download

DEFAULT_REPO = "leliuga/all-MiniLM-L6-v2-GGUF"
DEFAULT_TEST_MODEL = "all-MiniLM-L6-v2.Q4_0.gguf"


@pytest.fixture(scope="module")
def get_model() -> str:
    hf_hub_download(repo_id=DEFAULT_REPO, filename=DEFAULT_TEST_MODEL, local_dir="./local_model")
    return os.path.join("local_model", DEFAULT_TEST_MODEL)


def test_basic(get_model):
    # Using without a context manager
    embedder = LlamaEmbedder(get_model)
    embeddings = embedder.embed(["Hello, world!", "Another sentence"], norm=NormalizationType.EUCLIDEAN)
    assert embeddings is not None
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 384


def test_embed_err_no_texts(get_model):
    # Using without a context manager
    embedder = LlamaEmbedder(get_model)
    with pytest.raises(Exception) as e:
        embedder.embed([])
    assert str(e.value) == "Texts are empty"


def test_metadata(get_model):
    # Using without a context manager
    embedder = LlamaEmbedder(get_model)
    metadata_dict = embedder.get_metadata()
    assert metadata_dict is not None
    assert len(metadata_dict) > 0
    assert "general.name" in metadata_dict
    assert "general.architecture" in metadata_dict
    assert "bert.context_length" in metadata_dict
    assert metadata_dict["general.name"] == "all-MiniLM-L6-v2"
    assert metadata_dict["general.architecture"] == "bert"
    assert metadata_dict["bert.context_length"] == "512"


def get_attn_mask_len(attn_mask: List[int]) -> int:
    mask_size = 0
    for mask_t in attn_mask:
        if mask_t != 0:
            mask_size += 1

    return mask_size


def test_tokenize(get_model):
    # Using without a context manager
    embedder = LlamaEmbedder(get_model)
    tokenizer_data = embedder.tokenize(["Hello, world!", "How are you?"], enable_padding=True)
    assert tokenizer_data is not None
    assert len(tokenizer_data) == 2
    assert len(tokenizer_data[0].tokens) == 512
    assert len(tokenizer_data[0].attention_mask) == 512
    assert get_attn_mask_len(tokenizer_data[0].attention_mask) == 6
    assert len(tokenizer_data[1].tokens) == 512
    assert len(tokenizer_data[1].attention_mask) == 512
    assert get_attn_mask_len(tokenizer_data[1].attention_mask) == 6


def test_tokenize_without_special_tokens(get_model):
    # Using without a context manager
    embedder = LlamaEmbedder(get_model)
    tokenizer_data = embedder.tokenize(["Hello, world!", "How are you?"], add_special_tokens=False, enable_padding=True)
    assert tokenizer_data is not None
    assert len(tokenizer_data) == 2
    assert len(tokenizer_data[0].tokens) == 512
    assert len(tokenizer_data[0].attention_mask) == 512
    assert get_attn_mask_len(tokenizer_data[0].attention_mask) == 4
    assert len(tokenizer_data[1].tokens) == 512
    assert len(tokenizer_data[1].attention_mask) == 512
    assert get_attn_mask_len(tokenizer_data[1].attention_mask) == 4


def test_tokenize_err_no_texts(get_model):
    # Using without a context manager
    embedder = LlamaEmbedder(get_model)
    with pytest.raises(Exception) as e:
        embedder.tokenize([])
    assert str(e.value) == "Texts are empty"
