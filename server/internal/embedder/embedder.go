package embedder

/*
#cgo CFLAGS: -I. -I../../../src
#cgo CXXFLAGS: -I. -I../../../src -std=c++11 -Wall -Wextra -pedantic

#cgo LDFLAGS: -L../../../build/static -lllama-embedder -lcommon -lllama -lggml
#cgo darwin LDFLAGS: -stdlib=libc++ -framework Accelerate -framework Metal -framework Foundation -framework MetalKit
#cgo linux LDFLAGS: -lstdc++ -fopenmp
#cgo windows LDFLAGS: -lkernel32
#include <stdlib.h>
#include <stdio.h>
#include <wrapper.h>

*/
import "C"
import (
	"fmt"
	"github.com/amikos-tech/llamacpp-embedder/server/internal/utils"
	"os"
	"path/filepath"
	"sync"
	"unsafe"
)

type NormalizationType int32

type PoolingType int32

const (
	NormalizationNone        NormalizationType = -1
	NormalizationMaxAbsInt16 NormalizationType = 0
	NormalizationTaxicab     NormalizationType = 1
	NormalizationL2          NormalizationType = 2
	PoolingNone              PoolingType       = 0
	PoolingMean              PoolingType       = 1
	PoolingCls               PoolingType       = 2
	PoolingLast              PoolingType       = 3
)

type LlamaEmbedder struct {
	modelPath                string
	defaultNormalizationType NormalizationType
	defaultPoolingType       PoolingType
	hfRepo                   string
	localCacheDir            string
	embedder                 *C.llama_embedder
	mu                       sync.RWMutex
}

type Option func(*LlamaEmbedder) error

// WithNormalization sets the normalization type to use
// Possible values are NormalizationNone, NormalizationMaxAbsInt16, NormalizationTaxicab, NormalizationL2 (default)
func WithNormalization(norm NormalizationType) Option {
	return func(e *LlamaEmbedder) error {
		e.defaultNormalizationType = norm
		return nil
	}
}

// WithPooling sets the pooling type to use
// Possible values are PoolingNone, PoolingMean (default), PoolingCls, PoolingLast
func WithPooling(pool PoolingType) Option {
	return func(e *LlamaEmbedder) error {
		e.defaultPoolingType = pool
		return nil
	}
}

// WithHFRepo sets the Hugging Face repo to download the model from
func WithHFRepo(repo string) Option {
	return func(e *LlamaEmbedder) error {
		if repo == "" {
			return fmt.Errorf("HF repo is empty")
		}
		e.hfRepo = repo
		return nil
	}
}

// WithModelCacheDir sets the directory to cache the model. If the directory does not exist, it will be created.
func WithModelCacheDir(modelCacheDir string) Option {
	return func(e *LlamaEmbedder) error {
		if modelCacheDir == "" {
			return fmt.Errorf("model cache is not set")
		}
		if _, err := os.Stat(modelCacheDir); os.IsNotExist(err) {
			if err := os.MkdirAll(modelCacheDir, os.ModePerm); err != nil {
				return err
			}
		}
		absDir, err := filepath.Abs(modelCacheDir)
		if err != nil {
			return err
		}
		e.localCacheDir = absDir
		return nil
	}
}

func NewLlamaEmbedder(modelPath string, opts ...Option) (*LlamaEmbedder, func(), error) {
	e := &LlamaEmbedder{
		modelPath:                modelPath,
		defaultNormalizationType: NormalizationL2,
		defaultPoolingType:       PoolingMean,
	}
	for _, opt := range opts {
		if err := opt(e); err != nil {
			return nil, nil, err
		}
	}
	if modelPath == "" {
		return nil, nil, fmt.Errorf("modelPath is not set")
	}
	err := utils.EnsureCacheDir()
	if err != nil {
		return nil, nil, err
	}
	if e.hfRepo != "" {
		e.modelPath = filepath.Join(e.localCacheDir, filepath.Base(modelPath))
		err := utils.DownloadHFModel(e.hfRepo, modelPath, e.modelPath, "")
		if err != nil {
			return nil, nil, err
		}
	} else {
		if _, err := os.Stat(modelPath); os.IsNotExist(err) {
			return nil, nil, err
		}
		e.modelPath = modelPath
	}
	cModelPath := C.CString(modelPath)
	var embedder *C.llama_embedder
	result := C.init_embedder_l(&embedder, cModelPath, C.uint32_t(uint32(e.defaultPoolingType)))
	if result != 0 {
		return nil, nil, fmt.Errorf("failed to initialize embedder")
	}
	e.embedder = embedder
	return e, func() {
		e.Close()
		C.free(unsafe.Pointer(cModelPath))
	}, nil
}

// EmbedTexts embeds the given texts using the model
func (e *LlamaEmbedder) EmbedTexts(texts []string) ([][]float32, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()
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
	result := C.embed_texts(e.embedder, (**C.char)(unsafe.Pointer(&cTexts[0])), C.size_t(len(texts)), C.int32_t(int32(e.defaultNormalizationType)))
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

// Close closes the embedder and frees any resources
func (e *LlamaEmbedder) Close() {
	e.mu.RLock()
	defer e.mu.RUnlock()
	C.free_embedder_l(e.embedder)
}
