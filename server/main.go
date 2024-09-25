package main

/*
#cgo CFLAGS: -I. -I../src
#cgo CXXFLAGS: -I. -I../src -std=c++11 -Wall -Wextra -pedantic

#cgo LDFLAGS: -L../build/static -lllama-embedder -lcommon -lllama -lggml
#cgo darwin LDFLAGS: -stdlib=libc++ -framework Accelerate -framework Metal -framework Foundation -framework MetalKit
#cgo linux LDFLAGS: -ldl -lstdc++
#cgo windows LDFLAGS: -lkernel32
#include <stdlib.h>
#include <stdio.h>
#include <wrapper.h>

*/
import "C"
import (
	"fmt"
	"unsafe"
)

func main() {
	cModelPath := C.CString("/Users/tazarov/Downloads/all-MiniLM-L6-v2.F32.gguf")
	var embedder *C.llama_embedder
	result := C.init_embedder_l(&embedder, cModelPath, C.uint32_t(uint32(1)))
	if result != 0 {
		fmt.Printf("failed to initialize embedder")
	}
	embeddings, err := EmbedTexts(embedder, []string{"Hello world", "My name is Ishmael"})
	if err != nil {
		fmt.Printf("failed to embed text: %v", err)
	}
	fmt.Printf("Embeddings: %v", embeddings)
	C.free_embedder_l(embedder)
	defer C.free(unsafe.Pointer(cModelPath))
}

// EmbedTexts embeds the given texts using the model
func EmbedTexts(embedder *C.llama_embedder, texts []string) ([][]float32, error) {
	cTexts := make([]*C.char, len(texts))
	for i, t := range texts {
		cTexts[i] = C.CString(t)
	}
	// Free the C strings when we're done
	defer func() {
		for _, t := range cTexts {
			C.free(unsafe.Pointer(t))
		}
	}()
	result := C.embed_texts(embedder, (**C.char)(unsafe.Pointer(&cTexts[0])), C.size_t(len(texts)), C.int32_t(int32(2)))
	defer func() {
		C.free_float_matrixw(&result)
	}()
	if result.data == nil {
		return nil, fmt.Errorf("failed to embed text: %v", C.GoString(C.get_last_error()))
	}

	// Convert the result to a Go slice
	goResult := make([][]float32, result.rows)
	for i := 0; i < int(result.rows); i++ {
		goResult[i] = make([]float32, result.cols)
		for j := 0; j < int(result.cols); j++ {
			index := i*int(result.cols) + j
			goResult[i][j] = float32(*(*C.float)(unsafe.Pointer(uintptr(unsafe.Pointer(result.data)) + uintptr(index)*unsafe.Sizeof(C.float(0)))))
		}
	}
	return goResult, nil
}
