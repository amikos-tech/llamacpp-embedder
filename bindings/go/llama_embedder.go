package llama_embedder

/*
#cgo CFLAGS: -I.
#cgo CXXFLAGS: -I. -std=c++11 -Wall -Wextra -pedantic

#cgo darwin LDFLAGS: -stdlib=libc++
#cgo linux LDFLAGS: -ldl -lstdc++
#cgo windows LDFLAGS: -lkernel32
#include <stdlib.h>
#include "wrapper.h"

*/
import "C"
import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
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
	LatestSharedLibVersion                     = "v0.0.8"
)

type LlamaEmbedder struct {
	modelPath                    string
	sharedLibraryPath            string
	defaultNormalizationType     NormalizationType
	defaultPoolingType           PoolingType
	hfRepo                       string
	localCacheDir                string
	sharedLibraryVersion         string
	sharedLibPathUserProvided    bool
	sharedLibVersionUserProvided bool
}

type Option func(*LlamaEmbedder) error

var defaultCacheDir = filepath.Join(os.Getenv("HOME"), ".cache/llama_cache")
var defaultModelCacheDir = filepath.Join(defaultCacheDir, "models")
var defaultLibCacheDir = filepath.Join(defaultCacheDir, "libs")

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

// WithSharedLibraryVersion sets the shared library version to use. This is overridden by WithSharedLibraryPath
func WithSharedLibraryVersion(version string) Option {
	return func(e *LlamaEmbedder) error {
		if version == "" {
			return fmt.Errorf("shared library version is empty")
		}
		// TODO make the check more robust - latest or valid vX.Y.Z(-rcN/-alphaN/-betaN)
		e.sharedLibVersionUserProvided = true
		e.sharedLibraryVersion = version
		return nil
	}
}

// WithSharedLibraryPath sets the shared library path to use. LlamaEmbedder will look for shared library under this path.
// This overrides WithSharedLibraryVersion.
func WithSharedLibraryPath(libPath string) Option {
	return func(e *LlamaEmbedder) error {
		if libPath == "" {
			return fmt.Errorf("shared library path is not provided")
		}
		expandedPath, err := expandTilde(libPath)
		if err != nil {
			return err
		}
		if _, err := os.Stat(expandedPath); os.IsNotExist(err) {
			return fmt.Errorf("shared library path does not exist: %v", err)
		}
		e.sharedLibPathUserProvided = true
		e.sharedLibraryPath = expandedPath
		return nil
	}
}

func NewLlamaEmbedder(modelPath string, opts ...Option) (*LlamaEmbedder, func(), error) {
	e := &LlamaEmbedder{
		defaultNormalizationType: NormalizationL2,
		defaultPoolingType:       PoolingMean,
		localCacheDir:            defaultModelCacheDir,
		sharedLibraryPath:        filepath.Join(defaultLibCacheDir, LatestSharedLibVersion),
		sharedLibraryVersion:     LatestSharedLibVersion,
	}
	for _, opt := range opts {
		err := opt(e)
		if err != nil {
			return nil, nil, err
		}
	}
	if modelPath == "" {
		return nil, nil, fmt.Errorf("modelPath is not set")
	}
	err := ensureCacheDir()
	if err != nil {
		return nil, nil, err
	}
	if e.hfRepo != "" {
		e.modelPath = filepath.Join(e.localCacheDir, filepath.Base(modelPath))
		err := downloadHFModel(e.hfRepo, modelPath, e.modelPath, "")
		if err != nil {
			return nil, nil, err
		}
	} else {
		if _, err := os.Stat(modelPath); os.IsNotExist(err) {
			return nil, nil, err
		}
		e.modelPath = modelPath
	}
	err = e.loadLibrary()
	freeFunc := func() {
		e.Close()
	}
	if err != nil {

		return nil, nil, err
	}
	err = e.initEmbedder()
	if err != nil {
		defer freeFunc()
		return nil, nil, err
	}
	return e, freeFunc, nil
}

// loadLibrary loads the shared library.
// The method will first check if the user has provided a shared library path, then version and finally download the latest version.
func (e *LlamaEmbedder) loadLibrary() (err error) {
	var actualPath string
	if e.sharedLibPathUserProvided {
		actualPath = filepath.Join(e.sharedLibraryPath, getOSSharedLibName())
	} else if e.sharedLibVersionUserProvided {
		actualPath, err = ensureLibrary(e.sharedLibraryVersion)
		if err != nil {
			return
		}
	} else {
		actualPath, err = ensureLibrary(LatestSharedLibVersion)
		if err != nil {
			return
		}
	}
	if _, err := os.Stat(actualPath); os.IsNotExist(err) {
		return fmt.Errorf("shared library not found: %v", err)
	}
	cLibPath := C.CString(actualPath)
	defer C.free(unsafe.Pointer(cLibPath))
	if C.load_library(cLibPath) == nil {
		return fmt.Errorf("%v", C.GoString(C.get_last_error()))
	}
	return nil
}

// initEmbedder initializes the embedder with the given model
func (e *LlamaEmbedder) initEmbedder() error {
	cModelPath := C.CString(e.modelPath)
	defer C.free(unsafe.Pointer(cModelPath))
	if C.init_llama_embedder(cModelPath, C.uint32_t(uint32(e.defaultPoolingType))) != 0 {
		return fmt.Errorf("failed to initialize llama backend %v", C.GoString(C.get_last_error()))
	}
	return nil
}

// Close closes the embedder and frees any resources
func (e *LlamaEmbedder) Close() {
	C.free_llama_embedder()
}

// EmbedTexts embeds the given texts using the model
func (e *LlamaEmbedder) EmbedTexts(texts []string) ([][]float32, error) {
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
	result := C.llama_embedder_embed((**C.char)(unsafe.Pointer(&cTexts[0])), C.size_t(len(texts)), C.int32_t(int32(e.defaultNormalizationType)))
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

// GetMetadata returns the metadata associated with the model
func (e *LlamaEmbedder) GetMetadata() map[string]string {
	var size C.size_t
	cMetadataArray := C.llama_embedder_get_metadata(&size)
	defer C.free_metadata(cMetadataArray, size)

	metadata := make(map[string]string)
	for i := 0; i < int(size); i++ {
		entry := C.GoString(*(**C.char)(unsafe.Pointer(uintptr(unsafe.Pointer(cMetadataArray)) + uintptr(i)*unsafe.Sizeof(uintptr(0)))))
		parts := strings.SplitN(entry, "=", 2)
		if len(parts) == 2 {
			metadata[parts[0]] = parts[1]
		}
	}
	return metadata
}
