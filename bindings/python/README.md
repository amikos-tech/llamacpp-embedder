# Llama Embedder

This is a python binding for llama embedder, a purpose-built library for embeddings.

## Installation

```bash
pip install llama_embedder
```

## Usage

### Local Models

This example shows how to use local model to embed texts.

```python
from llama_embedder import Embedder

embedder = Embedder(model_path='./path/to/model.gguf')

# Embed stings

embeddings = embedder.embed_texts(["Hello World!", "My name is Ishmael."])
```

### Hugging Face Models

This example shows how to download and use a model from Hugging Face.

```python
from llama_embedder import Embedder

hf_repo = "ChristianAzinn/snowflake-arctic-embed-s-gguf"
gguf_file = "snowflake-arctic-embed-s-f16.GGUF"
embedder = Embedder(gguf_file, hf_repository=hf_repo)
embeddings = embedder.embed_texts(["Hello, world!", "Another sentence"])
```
