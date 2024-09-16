# Llama Embedder - Go Binding

## Installation

```bash
go get github.com/amikos-tech/llamacpp-embedder/bindings/go
```

## Usage


The following example shows how to use the go binding to load `snowflake-arctic-embed-s` model from HuggingFace.

> Note: As part of the initialization of the binding the model

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