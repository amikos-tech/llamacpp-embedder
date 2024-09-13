package llama_embedder

/*
#cgo CFLAGS: -I.
#cgo CXXFLAGS: -I. -std=c++11 -stdlib=libc++
#cgo LDFLAGS: -ldl -lc++
#include <stdlib.h>
#include "wrapper.h"

*/
import "C"
import (
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"unsafe"
)

type NormalizationType int32

type PoolingType int32

const (
	NORMALIZATION_NONE          NormalizationType = -1
	NORMALIZATION_MAX_ABS_INT16 NormalizationType = 0
	NORMALIZATION_TAXICAB       NormalizationType = 1
	NORMALIZATION_L2            NormalizationType = 2
	POOLING_NONE                PoolingType       = 0
	POOLING_MEAN                PoolingType       = 1
	POOLING_CLS                 PoolingType       = 2
	POOLING_LAST                PoolingType       = 3
)

type LlamaEmbedder struct {
	modelPath                string
	sharedLibraryPath        string
	defaultNormalizationType NormalizationType
	defaultPoolingType       PoolingType
	hfRepo                   string
	localCacheDir            string
}

type Option func(*LlamaEmbedder) error

var DefaultCacheDir = filepath.Join(os.Getenv("HOME"), ".cache/llama_cache")

func WithNormalization(norm NormalizationType) Option {
	return func(e *LlamaEmbedder) error {
		e.defaultNormalizationType = norm
		return nil
	}
}

func WithPooling(pool PoolingType) Option {
	return func(e *LlamaEmbedder) error {
		e.defaultPoolingType = pool
		return nil
	}
}

func WithHFRepo(repo string) Option {
	return func(e *LlamaEmbedder) error {
		if repo == "" {
			return fmt.Errorf("HF repo is empty")
		}
		e.hfRepo = repo
		return nil
	}
}

func WithModelCacheDir(dir string) Option {
	return func(e *LlamaEmbedder) error {
		if dir == "" {
			return fmt.Errorf("Model cache dir is empty")
		}
		if _, err := os.Stat(dir); os.IsNotExist(err) {
			if err := os.MkdirAll(dir, os.ModePerm); err != nil {
				return err
			}
		}
		absDir, err := filepath.Abs(dir)
		if err != nil {
			return err
		}
		e.localCacheDir = absDir
		return nil
	}
}

var defaults = []Option{
	WithNormalization(NORMALIZATION_L2),
	WithPooling(POOLING_MEAN),
	WithModelCacheDir(DefaultCacheDir),
}

func NewLlamaEmbedder(sharedLibraryPath string, modelPath string, opts ...Option) (*LlamaEmbedder, func(), error) {
	e := &LlamaEmbedder{sharedLibraryPath: sharedLibraryPath}

	for _, opt := range append(defaults, opts...) {
		err := opt(e)
		if err != nil {
			return nil, nil, err
		}
	}
	if sharedLibraryPath == "" {
		return nil, nil, fmt.Errorf("sharedLibraryPath is not set")
	}

	if _, err := os.Stat(sharedLibraryPath); os.IsNotExist(err) {
		return nil, nil, err
	}
	if modelPath == "" {
		return nil, nil, fmt.Errorf("modelPath is not set")
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
	err := e.loadLibrary()
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

// loadLibrary loads the shared library
func (e *LlamaEmbedder) loadLibrary() error {
	cLibPath := C.CString(e.sharedLibraryPath)
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
func (e *LlamaEmbedder) EmbedTexts(texts []string) [][]float32 {
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
		C.free_float_matrix(result)
	}()

	// Convert the result to a Go slice
	goResult := make([][]float32, result.rows)
	for i := 0; i < int(result.rows); i++ {
		goResult[i] = make([]float32, result.cols)
		for j := 0; j < int(result.cols); j++ {
			index := i*int(result.cols) + j
			goResult[i][j] = float32(*(*C.float)(unsafe.Pointer(uintptr(unsafe.Pointer(result.data)) + uintptr(index)*unsafe.Sizeof(C.float(0)))))
		}
	}
	return goResult
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

func downloadHFModel(hfRepo, hfFile, targetLocation, hfToken string) error {
	if _, err := os.Stat(targetLocation); err == nil {
		return nil
	}
	client := &http.Client{}

	url := fmt.Sprintf("https://huggingface.co/%s/resolve/main/%s", hfRepo, hfFile)

	// Create HTTP GET request
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return err
	}

	// Set Authorization header if HF_TOKEN is provided
	if hfToken != "" {
		req.Header.Set("Authorization", "Bearer "+hfToken)
	}

	// Execute the request
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	// Check for HTTP errors
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("HTTP error: %s", resp.Status)
	}

	// Extract filename from URL
	segments := strings.Split(url, "/")
	filename := segments[len(segments)-1]
	if filename == "" {
		filename = "model.gguf"
	}

	var outputPath string

	// Determine the output path based on targetLocation
	if targetLocation != "" {
		// Check if targetLocation is a directory
		info, err := os.Stat(targetLocation)
		if err == nil && info.IsDir() {
			// targetLocation is an existing directory
			outputPath = filepath.Join(targetLocation, filename)
		} else if os.IsNotExist(err) && strings.HasSuffix(targetLocation, string(os.PathSeparator)) {
			// targetLocation is a non-existing directory (ends with / or \)
			if err := os.MkdirAll(targetLocation, os.ModePerm); err != nil {
				return fmt.Errorf("failed to create directory: %v", err)
			}
			outputPath = filepath.Join(targetLocation, filename)
		} else {
			// targetLocation is a file path
			outputDir := filepath.Dir(targetLocation)
			// Create the directory if it doesn't exist
			if err := os.MkdirAll(outputDir, os.ModePerm); err != nil {
				return fmt.Errorf("failed to create directory: %v", err)
			}
			outputPath = targetLocation
		}
	} else {
		// No targetLocation provided, use current directory
		outputPath = filename
	}

	// Create the output file
	outFile, err := os.Create(outputPath)
	if err != nil {
		return err
	}
	defer outFile.Close()

	// Write response body to file
	_, err = io.Copy(outFile, resp.Body)
	return err
}

func getOSSharedLibName() string {
	switch cos := strings.ToLower(runtime.GOOS); cos {
	case "darwin":
		return "libllama-embedder.dylib"
	case "windows":
		return "libllama-embedder.dll"
	default:
		return "libllama-embedder.so"
	}
}
