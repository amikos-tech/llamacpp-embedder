# Llama.cpp Embedder Library

The goal of this library is to deliver good developer experience for users that need to generate embeddings
using [Llama.cpp](https://github.com/ggerganov/llama.cpp).

For now the library is built for maximum CPU compatibility without any AVX or other SIMD optimizations.

This library builds a shared lib module that can be used with the various bindings to creat embedding functions that run
locally.

⚠️ For now we distribute only binaries and while we try to build for most platforms it is our intention to also deliver
source distributable that can be built on target platform. Building from source is far less user-friendly and is
intended for advanced users that want a custom builds e.g. for GPU support.

## Bindings

The usefulness of llama-embedder library lies in the bindings that allow it to be used in various languages.

- [Python](#python) - self-contained python binary package shipped for Linux (x86_64, arm64), MacOS (x86_64, arm64) and
  Windows (x86_64)
- [Golang](#golang) - go module that can be integrated in your go project with no additional dependencies. The module
  has builtin mechanism to get the appropriate shared library for the platform.
- [Node.js](#nodejs) - coming soon
- [Java](#java) - coming soon

### Python

The library comes in zero-dependency small python package - https://pypi.org/project/llama-embedder/

**Installation:**

```bash
pip install llama-embedder
```

**Usage - Local Models:**

```python
from llama_embedder import Embedder

embedder = Embedder(model_path='./path/to/model.gguf')

# Embed stings

embeddings = embedder.embed_texts(["Hello World!", "My name is Ishmael."])
```

**Usage - Hugging Face Models:**

```python
from llama_embedder import Embedder

hf_repo = "ChristianAzinn/snowflake-arctic-embed-s-gguf"
gguf_file = "snowflake-arctic-embed-s-f16.GGUF"
embedder = Embedder(gguf_file, hf_repository=hf_repo)
embeddings = embedder.embed_texts(["Hello, world!", "My name is Ishmael."])
```

### Golang

The library is also available as a Go module - https://pkg.go.dev/github.com/amikos-tech/llamacpp-embedder/bindings/go

**Installation:**

```bash
go get github.com/amikos-tech/llamacpp-embedder/bindings/go
```

**Usage:**

```go
package main

import (
	"fmt"
	llama "github.com/amikos-tech/llamacpp-embedder/bindings/go"
)

func main() {
	hfRepo := "ChristianAzinn/snowflake-arctic-embed-s-gguf"
	hfFile := "snowflake-arctic-embed-s-f16.GGUF"
	e, closeFunc, err := llama.NewLlamaEmbedder(hfFile, llama.WithHFRepo(hfRepo))
	if err != nil {
        panic(err)
    }
	defer closeFunc()
	res, err := e.EmbedTexts([]string{"Hello world", "My name is Ishmael"})
    if err != nil {
        panic(err)
    }
	
    for _, r := range res {
        fmt.Println(r)
    }
}
```

### Node.js

Coming soon. If you are interested and would like to contribute, we invite you to submit a PR.

### Java

Coming soon. If you are interested and would like to contribute, we invite you to submit a PR.

## Building the library

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