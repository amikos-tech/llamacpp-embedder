package llama_embedder

import (
	"fmt"
	"github.com/stretchr/testify/require"
	"os"
	"path/filepath"
	"testing"
)

func TestDownloadVersion(t *testing.T) {
	libFilePath, err := ensureLibrary("v0.0.8")
	parent := filepath.Dir(libFilePath)
	require.NoError(t, err, "Error downloading version: %v", err)
	require.FileExists(t, libFilePath, "Library file should exist")
	t.Cleanup(func() {
		// remove the parent directory
		err := os.RemoveAll(parent)
		if err != nil {
			fmt.Printf("Error removing parent directory: %v\n", err)
		}
	})
}

func TestDownloadModel(t *testing.T) {
	err := downloadHFModel(defaultHFRepo, defaultModelFile, defaultModelFile, "")
	require.NoError(t, err, "Failed to download model")
	_, err = os.Stat(defaultModelFile)
	require.NoError(t, err, "Failed to download model")
}
