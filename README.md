# Llama.cpp Embedder Library

The goal of this library is to deliver good developer experience for users that need to generate embeddings
using [Llama.cpp](https://github.com/ggerganov/llama.cpp).

For now the library is built for maximum CPU compatibility without any AVX or other SIMD optimizations.

This library builds a shared lib module that can be used with the various bindings to creat embedding functions that run
locally.

⚠️ For now we distribute only binaries and while we try to build for most platforms it is our intention to also deliver
source distributable that can be built on target platform. Building from source is far less user-friendly and is
intended for advanced users that want a custom builds e.g. for GPU support.

## Building

This project requires cmake to build.

### Shared library

To build the shared library run:

```bash
make lib
```

To run the tests:

```bash
make lib-test
```

### Python bindings

To build the python bindings run:

```bash
python -m venv venv
source venv/bin/activate
make python-cidist-local
```

The above command will build the shared library, the python binding package and run the tests.
