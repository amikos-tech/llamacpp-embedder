# Llama Embedder Server

A simple server to serve embedding inference using llama.cpp models (GGUF).

## Up and Running

```bash
go install github.com/amikos-tech/llamacpp-embedder/server
export LLAMA_CACHE_DIR=./cache # set model cache directory
export LLAMA_MODEL_TTL_MINUTES=60 # 1 hour
export LLAMA_CACHED_MODELS="ChristianAzinn/snowflake-arctic-embed-s-gguf/snowflake-arctic-embed-s-f16.GGUF;leliuga/all-MiniLM-L6-v2-GGUF/all-MiniLM-L6-v2.Q4_0.gguf" # list of models to cache (requires internet connection)
llama-embedder-server
```

### Endpoints

- `/embed_texts` - POST - Embed a list of texts
- `/embed_models` - GET - List of cached models
- `/version` - GET - Server version
- `/health` - GET - Server health

### Environment Variables

- `LLAMA_CACHE_DIR` - Directory to cache models (default: `~/./cache/llama_cache`)
- `LLAMA_MODEL_TTL_MINUTES` - TTL for cached models in minutes (default: `60`)
- `LLAMA_CACHED_MODELS` - List of models to cache. If the models are not cached, the server will download them from Hugging Face Hub.
