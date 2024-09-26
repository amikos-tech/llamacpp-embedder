package embedder

import (
	"github.com/amikos-tech/llamacpp-embedder/server/internal/utils"
	"github.com/stretchr/testify/require"
	"os"
	"path/filepath"
	"testing"
)

const defaultHFRepo = "leliuga/all-MiniLM-L6-v2-GGUF"
const defaultModelFile = "all-MiniLM-L6-v2.Q4_0.gguf"

func ensureModel(t *testing.T) string {
	tempDir := t.TempDir()
	targetLocation := filepath.Join(tempDir, "llama_cache", defaultModelFile)
	err := utils.DownloadHFModel(defaultHFRepo, defaultModelFile, targetLocation, "")
	require.NoErrorf(t, err, "Failed to download model: %v", err)
	return targetLocation
}

func TestNewLlamaEmbedder(t *testing.T) {
	// Test with default options

	t.Run("DefaultOptions", func(t *testing.T) {
		modelPath := ensureModel(t)
		embedder, cleanup, err := NewLlamaEmbedder(modelPath)
		require.NoErrorf(t, err, "expected no error, got %v", err)
		t.Cleanup(cleanup)

		require.Equalf(t, modelPath, embedder.modelPath, "expected modelPath %v, got %v", modelPath, embedder.modelPath)
		require.Equalf(t, NormalizationL2, embedder.defaultNormalizationType, "expected defaultNormalizationType %v, got %v", NormalizationL2, embedder.defaultNormalizationType)
		require.Equalf(t, PoolingMean, embedder.defaultPoolingType, "expected defaultPoolingType %v, got %v", PoolingMean, embedder.defaultPoolingType)
	})

	// Test with custom normalization
	t.Run("WithNormalization", func(t *testing.T) {
		modelPath := ensureModel(t)
		embedder, cleanup, err := NewLlamaEmbedder(modelPath, WithNormalization(NormalizationMaxAbsInt16))
		require.NoErrorf(t, err, "expected no error, got %v", err)
		t.Cleanup(cleanup)

		require.Equalf(t, NormalizationMaxAbsInt16, embedder.defaultNormalizationType, "expected defaultNormalizationType to not be %v", NormalizationMaxAbsInt16)

	})

	// Test with custom pooling
	t.Run("WithPooling", func(t *testing.T) {
		modelPath := ensureModel(t)
		embedder, cleanup, err := NewLlamaEmbedder(modelPath, WithPooling(PoolingCls))
		require.NoErrorf(t, err, "expected no error, got %v", err)
		t.Cleanup(cleanup)
		require.Equalf(t, PoolingCls, embedder.defaultPoolingType, "expected defaultPoolingType to not be %v", PoolingCls)
	})

	// Test with Hugging Face repo
	t.Run("WithHFRepo", func(t *testing.T) {
		embedder, cleanup, err := NewLlamaEmbedder(defaultModelFile, WithHFRepo(defaultHFRepo))
		require.NoErrorf(t, err, "expected no error, got %v", err)
		defer cleanup()

		require.Equalf(t, defaultHFRepo, embedder.hfRepo, "expected hfRepo %v, got %v", defaultHFRepo, embedder.hfRepo)
	})

	// Test with model cache directory
	t.Run("WithModelCacheDir", func(t *testing.T) {
		modelPath := ensureModel(t)
		cacheDir := t.TempDir()
		embedder, cleanup, err := NewLlamaEmbedder(modelPath, WithModelCacheDir(cacheDir))
		require.NoErrorf(t, err, "expected no error, got %v", err)

		absCacheDir, _ := filepath.Abs(cacheDir)
		require.Equalf(t, absCacheDir, embedder.localCacheDir, "expected localCacheDir %v, got %v", absCacheDir, embedder.localCacheDir)

		// Clean up the created directory
		t.Cleanup(func() {
			cleanup()
			err := os.RemoveAll(cacheDir)
			if err != nil {
				t.Logf("Error removing cache dir: %v\n", err)
			}
		})
	})
}

func TestEmbedTexts(t *testing.T) {
	modelPath := ensureModel(t)
	embedder, cleanup, err := NewLlamaEmbedder(modelPath)
	require.NoErrorf(t, err, "expected no error, got %v", err)
	require.NoError(t, err)
	t.Cleanup(cleanup)

	texts := []string{"Hello, world!", "This is a test."}
	embeddings, err := embedder.EmbedTexts(texts)
	require.NoError(t, err)
	require.NotNil(t, embeddings)
	require.Equal(t, len(texts), len(embeddings))
	for _, embedding := range embeddings {
		require.NotEmpty(t, embedding)
		require.Len(t, embedding, 384)
	}
}
