package api

import (
	"bytes"
	"encoding/json"
	"github.com/amikos-tech/llamacpp-embedder/server/internal/middleware"
	"github.com/amikos-tech/llamacpp-embedder/server/internal/types"
	"github.com/amikos-tech/llamacpp-embedder/server/internal/utils"
	"github.com/stretchr/testify/require"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"testing"
)

const defaultHFRepo = "leliuga/all-MiniLM-L6-v2-GGUF"
const defaultModelFile = "all-MiniLM-L6-v2.Q4_0.gguf"

func TestHealthHandler(t *testing.T) {
	req, err := http.NewRequest("GET", "/health", nil)
	if err != nil {
		t.Fatal(err)
	}

	rr := httptest.NewRecorder()
	handler := http.HandlerFunc(HealthHandler)

	handler.ServeHTTP(rr, req)

	if status := rr.Code; status != http.StatusOK {
		t.Errorf("handler returned wrong status code: got %v want %v",
			status, http.StatusOK)
	}

	var returned map[string]any

	err = json.Unmarshal(rr.Body.Bytes(), &returned)
	require.NoError(t, err, "Failed to unmarshal response")
	require.Contains(t, returned, "status")
	require.Equal(t, "running", returned["status"])
	require.Contains(t, returned, "time")
	require.IsTypef(t, float64(0), returned["time"], "time should be a float64")
}

func TestVersionHandler(t *testing.T) {
	req, err := http.NewRequest("GET", "/version", nil)
	if err != nil {
		t.Fatal(err)
	}

	rr := httptest.NewRecorder()
	handler := http.HandlerFunc(VersionHandler)

	handler.ServeHTTP(rr, req)

	if status := rr.Code; status != http.StatusOK {
		t.Errorf("handler returned wrong status code: got %v want %v",
			status, http.StatusOK)
	}

	var returned map[string]any

	err = json.Unmarshal(rr.Body.Bytes(), &returned)
	require.NoError(t, err, "Failed to unmarshal response")
	require.Contains(t, returned, "version")
	require.Equal(t, types.VERSION, returned["version"])
}

func TestEmbedModelsHandler(t *testing.T) {
	err := utils.EnsureCacheDir()
	require.NoErrorf(t, err, "Error creating cache directory: %v", err)
	err = utils.DownloadHFModel(defaultHFRepo, defaultModelFile, filepath.Join(utils.GetModelCacheDir(), defaultModelFile), "")
	require.NoError(t, err, "Failed to download model")
	req, err := http.NewRequest("GET", "/embed_models", nil)
	require.NoError(t, err, "Failed to create request")
	rr := httptest.NewRecorder()
	handler := http.HandlerFunc(EmbedModelsHandler)
	handler.ServeHTTP(rr, req)

	if status := rr.Code; status != http.StatusOK {
		t.Errorf("handler returned wrong status code: got %v want %v",
			status, http.StatusOK)
	}
	var returned map[string][]string
	err = json.Unmarshal(rr.Body.Bytes(), &returned)
	require.NoError(t, err, "Failed to unmarshal response")
	require.Contains(t, returned, "models")
	require.IsType(t, []string{}, returned["models"])
	require.Contains(t, returned["models"], defaultModelFile)
}

func TestEmbedTextsHandler(t *testing.T) {
	err := utils.EnsureCacheDir()
	require.NoErrorf(t, err, "Error creating cache directory: %v", err)
	err = utils.DownloadHFModel(defaultHFRepo, defaultModelFile, filepath.Join(utils.GetModelCacheDir(), defaultModelFile), "")
	require.NoError(t, err, "Failed to download model")
	embedReq := types.EmbedRequest{Model: defaultModelFile, Texts: []string{"hello", "world"}}
	marshal, err := json.Marshal(embedReq)
	require.NoError(t, err, "Failed to marshal request")
	req, err := http.NewRequest("POST", "/embed_texts", bytes.NewBuffer(marshal))
	require.NoError(t, err, "Failed to create request")
	rr := httptest.NewRecorder()
	handler := http.Handler(middleware.CachingMiddleware(http.HandlerFunc(EmbedTextsHandler)))
	handler.ServeHTTP(rr, req)

	if status := rr.Code; status != http.StatusOK {
		t.Errorf("handler returned wrong status code: got %v want %v",
			status, http.StatusOK)
	}
	var returned types.EmbedResponse
	err = json.Unmarshal(rr.Body.Bytes(), &returned)
	require.NoError(t, err, "Failed to unmarshal response")
	require.NotNil(t, returned.Embeddings, "Embeddings should not be nil")
	require.Len(t, returned.Embeddings, 2, "Embeddings should have length 2")
	for _, r := range returned.Embeddings {
		require.Len(t, r, 384, "Embeddings should have length 384")
	}
}
