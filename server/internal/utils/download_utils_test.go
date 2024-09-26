package utils

import (
	"fmt"
	"github.com/stretchr/testify/require"
	"os"
	"path/filepath"
	"testing"
)

const defaultHFRepo = "leliuga/all-MiniLM-L6-v2-GGUF"
const defaultModelFile = "all-MiniLM-L6-v2.Q4_0.gguf"

func TestDownloadModel(t *testing.T) {
	tempDir := t.TempDir()
	targetLocation := filepath.Join(tempDir, "llama_cache", defaultModelFile)
	err := DownloadHFModel(defaultHFRepo, defaultModelFile, targetLocation, "")
	require.NoError(t, err, "Failed to download model")
	_, err = os.Stat(targetLocation)
	require.NoError(t, err, "Failed to download model")
	t.Cleanup(func() {
		err := os.RemoveAll(targetLocation)
		if err != nil {
			fmt.Printf("Error removing file: %v", err)
		}
	})
}

func TestEnsureSingleModel(t *testing.T) {
	tempDir := t.TempDir()
	models := fmt.Sprintf("%s/%s", defaultHFRepo, defaultModelFile)
	defaultModelCacheDir = filepath.Join(tempDir, "llama_cache", "models")
	modelPath := filepath.Join(defaultModelCacheDir, defaultModelFile)
	err := EnsureModels(models)
	require.NoError(t, err, "Failed to download model")
	_, err = os.Stat(modelPath)
	require.NoError(t, err, "Failed to download model")
	t.Cleanup(func() {
		err := os.RemoveAll(modelPath)
		if err != nil {
			fmt.Printf("Error removing file: %v", err)
		}
	})
}

func TestEnsureMultipleModels(t *testing.T) {
	tempDir := t.TempDir()
	models := fmt.Sprintf("%s/%s;%s/%s", defaultHFRepo, defaultModelFile, "ChristianAzinn/snowflake-arctic-embed-s-gguf", "snowflake-arctic-embed-s-f16.GGUF")
	defaultModelCacheDir = filepath.Join(tempDir, "llama_cache", "models")
	modelPaths := []string{
		filepath.Join(defaultModelCacheDir, defaultModelFile),
		filepath.Join(defaultModelCacheDir, "snowflake-arctic-embed-s-f16.GGUF"),
	}
	err := EnsureModels(models)
	require.NoError(t, err, "Failed to download model")
	for _, modelPath := range modelPaths {
		_, err = os.Stat(modelPath)
		require.NoError(t, err, "Failed to download model")
	}
	t.Cleanup(func() {
		err := os.RemoveAll(defaultModelCacheDir)
		if err != nil {
			fmt.Printf("Error removing file: %v", err)
		}
	})
}
