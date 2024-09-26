package utils

import (
	"os"
	"path/filepath"
	"testing"
)

func TestEnsureCacheDir(t *testing.T) {
	// Setup: create a temporary directory
	tempDir := t.TempDir()
	originalCacheDir := defaultCacheDir
	originalModelCacheDir := defaultModelCacheDir

	// Override the default cache directories with the temporary directory
	defaultCacheDir = filepath.Join(tempDir, "llama_cache")
	defaultModelCacheDir = filepath.Join(defaultCacheDir, "models")

	// Ensure to reset the default directories after the test
	t.Cleanup(func() {
		defaultCacheDir = originalCacheDir
		defaultModelCacheDir = originalModelCacheDir
	})

	// Call the function
	err := EnsureCacheDir()
	if err != nil {
		t.Fatalf("ensureCacheDir() failed: %v", err)
	}

	// Check if the directories were created
	if _, err := os.Stat(defaultCacheDir); os.IsNotExist(err) {
		t.Errorf("Cache directory was not created: %v", defaultCacheDir)
	}
	if _, err := os.Stat(defaultModelCacheDir); os.IsNotExist(err) {
		t.Errorf("Model cache directory was not created: %v", defaultModelCacheDir)
	}
}
