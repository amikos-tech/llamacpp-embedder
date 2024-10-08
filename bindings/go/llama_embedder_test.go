package llama_embedder

import (
	"fmt"
	"github.com/stretchr/testify/require"
	"os"
	"testing"
)

const defaultHFRepo = "leliuga/all-MiniLM-L6-v2-GGUF"
const defaultModelFile = "all-MiniLM-L6-v2.Q4_0.gguf"

func TestLlamaEmbedder(t *testing.T) {
	sharedLibPath := os.Getenv("SHARED_LIB_PATH")
	if sharedLibPath == "" {
		sharedLibPath = "../../build/"
	}
	err := downloadHFModel(defaultHFRepo, defaultModelFile, defaultModelFile, "")
	require.NoError(t, err, "Failed to download model")
	t.Cleanup(func() {
		err := os.Remove("all-MiniLM-L6-v2.Q4_0.gguf")
		if err != nil {
			fmt.Printf("Error removing file: %v", err)
		}
	})
	t.Run("Test Init", func(t *testing.T) {
		_, closeFunc, err := NewLlamaEmbedder(defaultModelFile, WithSharedLibraryPath(sharedLibPath))
		require.NoError(t, err, "Failed to create LlamaEmbedder")
		t.Cleanup(closeFunc)
	})
	t.Run("Test EmbedTexts", func(t *testing.T) {
		e, closeFunc, err := NewLlamaEmbedder(defaultModelFile, WithSharedLibraryPath(sharedLibPath))
		require.NoError(t, err, "Failed to create LlamaEmbedder")
		t.Cleanup(closeFunc)

		res, err := e.EmbedTexts([]string{"hello", "world"})
		require.NoError(t, err, "Failed to embed texts")
		require.Len(t, res, 2, "Failed to embed texts")
		for _, r := range res {
			require.Len(t, r, 384, "Failed to embed texts")
		}
	})

	t.Run("Test GetMetadata", func(t *testing.T) {
		e, closeFunc, err := NewLlamaEmbedder(defaultModelFile, WithSharedLibraryPath(sharedLibPath))
		require.NoError(t, err, "Failed to create LlamaEmbedder")
		t.Cleanup(closeFunc)

		metadata := e.GetMetadata()
		require.Greater(t, len(metadata), 0, "Failed to get metadata")
		require.Contains(t, metadata, "general.name", "Failed to get metadata")
		require.Contains(t, metadata, "general.architecture", "Failed to get metadata")
		require.Equal(t, "all-MiniLM-L6-v2", metadata["general.name"], "Failed to get metadata")
		require.Equal(t, "bert", metadata["general.architecture"], "Failed to get metadata")
	})

	t.Run("Test With HF Model", func(t *testing.T) {
		hfRepo := "ChristianAzinn/snowflake-arctic-embed-s-gguf"
		hfFile := "snowflake-arctic-embed-s-f16.GGUF"
		e, closeFunc, err := NewLlamaEmbedder(hfFile, WithSharedLibraryPath(sharedLibPath), WithHFRepo(hfRepo))
		require.NoError(t, err, "Failed to create LlamaEmbedder")
		t.Cleanup(func() {
			closeFunc()
		})
		res, err := e.EmbedTexts([]string{"hello", "world"})
		require.NoError(t, err, "Failed to embed texts")
		require.Len(t, res, 2, "Failed to embed texts")
		for _, r := range res {
			require.Len(t, r, 384, "Failed to embed texts")
		}
	})

	t.Run("Test With Download Shared libs", func(t *testing.T) {
		e, closeFunc, err := NewLlamaEmbedder(defaultModelFile, WithSharedLibraryVersion(LatestSharedLibVersion))
		require.NoError(t, err, "Failed to create LlamaEmbedder")
		t.Cleanup(func() {
			closeFunc()
		})
		res, err := e.EmbedTexts([]string{"hello", "world"})
		require.NoError(t, err, "Failed to embed texts")
		require.Len(t, res, 2, "Failed to embed texts")
		for _, r := range res {
			require.Len(t, r, 384, "Failed to embed texts")
		}
	})

	t.Run("Test with HF Model and target cache dir", func(t *testing.T) {
		hfRepo := "ChristianAzinn/snowflake-arctic-embed-s-gguf"
		hfFile := "snowflake-arctic-embed-s-f16.GGUF"
		e, closeFunc, err := NewLlamaEmbedder(hfFile, WithSharedLibraryPath(sharedLibPath), WithHFRepo(hfRepo), WithModelCacheDir("./cache"))
		require.NoError(t, err, "Failed to create LlamaEmbedder")

		if _, err := os.Stat("./cache/snowflake-arctic-embed-s-f16.GGUF"); os.IsNotExist(err) {
			require.NoError(t, err, "Failed to create cache dir")
		}
		t.Cleanup(func() {
			closeFunc()
			err := os.RemoveAll("./cache")
			if err != nil {
				fmt.Printf("Error removing cache dir: %v\n", err)
			}
		})
		res, err := e.EmbedTexts([]string{"hello", "world"})
		require.NoError(t, err, "Failed to embed texts")
		require.Len(t, res, 2, "Failed to embed texts")
		for _, r := range res {
			require.Len(t, r, 384, "Failed to embed texts")
		}
	})

}
