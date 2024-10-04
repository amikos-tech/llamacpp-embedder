package api

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	cache2 "github.com/amikos-tech/llamacpp-embedder/server/internal/cache"
	"github.com/amikos-tech/llamacpp-embedder/server/internal/types"
	"github.com/amikos-tech/llamacpp-embedder/server/internal/utils"
	"github.com/amikos-tech/llamacpp-embedder/server/internal/worker"
)

func EmbedModelsHandler(w http.ResponseWriter, _ *http.Request) {
	files, err := os.ReadDir(utils.GetModelCacheDir())
	if err != nil {
		http.Error(w, "Failed to read cache directory", http.StatusInternalServerError)
		return
	}

	var ggufFiles []string
	for _, file := range files {
		if !file.IsDir() && filepath.Ext(file.Name()) == ".gguf" {
			ggufFiles = append(ggufFiles, file.Name())
		}
	}

	resp := types.EmbedModelListResponse{Models: ggufFiles}
	w.Header().Set("Content-Type", "application/json")
	err = json.NewEncoder(w).Encode(resp)
	if err != nil {
		http.Error(w, "Failed to encode response", http.StatusInternalServerError)
		return
	}
}

func EmbedTextsHandler(w http.ResponseWriter, r *http.Request) {
	var req types.EmbedRequest
	err := json.NewDecoder(r.Body).Decode(&req)
	if err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	if !strings.HasSuffix(strings.ToLower(req.Model), ".gguf") || strings.Contains(req.Model, "/") || strings.Contains(req.Model, "\\") || strings.Contains(req.Model, "..") {
		http.Error(w, "Invalid model", http.StatusBadRequest)
		return
	}

	cache := r.Context().Value("cache").(*cache2.Cache)
	if cache == nil {
		http.Error(w, "Cache not found", http.StatusInternalServerError)
		return
	}

	pool, err := cache.GetOrCreateWorkerPool(req.Model, 5) // Create a pool with 5 workers
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to get or create worker pool: %v", err), http.StatusInternalServerError)
		return
	}

	responseChan := make(chan *types.EmbedResponse)
	job := worker.Job{
		Request:  &req,
		Response: responseChan,
	}

	pool.Submit(job)
	resp := <-responseChan

	if resp.Error != "" {
		http.Error(w, resp.Error, http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	err = json.NewEncoder(w).Encode(resp)
	if err != nil {
		http.Error(w, "Failed to encode response", http.StatusInternalServerError)
		return
	}
}

func VersionHandler(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	err := json.NewEncoder(w).Encode(map[string]string{"version": types.VERSION})
	if err != nil {
		http.Error(w, "Failed to encode response", http.StatusInternalServerError)
		return
	}
}

func HealthHandler(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	err := json.NewEncoder(w).Encode(map[string]any{"status": "running", "time": time.Now().Unix()})
	if err != nil {
		http.Error(w, "Failed to encode response", http.StatusInternalServerError)
		return
	}
}
