# Llama Embedder

This is a python binding for llama embedder, a purpose-built library for embeddings.

## Installation

```bash
pip install llama_embedder
```

## Usage

```python
from llama_embedder import LlamaEmbedder

embedder = LlamaEmbedder(model_path='./path/to/model.gguf')

# Embed stings

embeddings = embedder.embed(['Hello World!', 'My name is Ishmael.'])
```
