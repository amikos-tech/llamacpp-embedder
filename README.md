# Llama.cpp Embedder Library

The goal of this library is to deliver good developer experience for users that need to generate embeddings
using [Llama.cpp](https://github.com/ggerganov/llama.cpp).

This library builds a shared lib module that can be used with the various bindings to creat embedding functions that run
locally.

## Building

This project requires cmake to build.

```bash
sysctl -a
mkdir build
cd build
cmake -DLLAMA_FATAL_WARNINGS=ON -DGGML_METAL_EMBED_LIBRARY=ON -DLLAMA_CURL=ON -DGGML_RPC=ON -DBUILD_SHARED_LIBS=ON ..
cmake --build . --config Release -j $(sysctl -n hw.logicalcpu)
```
