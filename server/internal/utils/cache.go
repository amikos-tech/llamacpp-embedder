package utils

import (
	"fmt"
	"os"
	"path/filepath"
)

var defaultCacheDir = filepath.Join(os.Getenv("HOME"), ".cache/llama_cache")
var defaultModelCacheDir = filepath.Join(defaultCacheDir, "models")

func EnsureCacheDir() error {
	if _, err := os.Stat(defaultCacheDir); os.IsNotExist(err) {
		err := os.MkdirAll(defaultCacheDir, 0755)
		if err != nil {
			return fmt.Errorf("could not create cache directory: %v", err)
		}
		err = os.MkdirAll(defaultModelCacheDir, 0755)
		if err != nil {
			return fmt.Errorf("could not create model cache directory: %v", err)
		}
	}
	return nil
}

func GetCacheDir() string {
	return defaultCacheDir
}

func GetModelCacheDir() string {
	return defaultModelCacheDir
}

func init() {
	if envProvidedCacheDir, exists := os.LookupEnv("LLAMA_CACHE_DIR"); exists {
		defaultCacheDir = envProvidedCacheDir
		defaultModelCacheDir = filepath.Join(defaultCacheDir, "models")
	}
	err := EnsureCacheDir()
	if err != nil {
		panic(fmt.Sprintf("Error creating cache directory: %v", err))
	}
}
