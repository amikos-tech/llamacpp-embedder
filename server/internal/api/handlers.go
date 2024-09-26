package api

import (
	"encoding/json"
	cache2 "github.com/amikos-tech/llamacpp-embedder/server/internal/cache"
	"github.com/amikos-tech/llamacpp-embedder/server/internal/embedder"
	"github.com/amikos-tech/llamacpp-embedder/server/internal/types"
	"github.com/amikos-tech/llamacpp-embedder/server/internal/utils"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"time"
)

func EmbedModelsHandler(w http.ResponseWriter, r *http.Request) {
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

	resp := types.EmbedModelList{Models: ggufFiles}
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
	cache := r.Context().Value("cache").(*cache2.Cache)
	if cache == nil {
		http.Error(w, "Cache not found", http.StatusInternalServerError)
		return
	}
	var embdr *embedder.LlamaEmbedder
	if e, found := cache.Get(req.Model); found {
		embdr = e
	} else {
		embdr, _, err = embedder.NewLlamaEmbedder(filepath.Join(utils.GetModelCacheDir(), req.Model))
		if err != nil {
			http.Error(w, "Failed to create embedder", http.StatusInternalServerError)
			return
		}
		//TODO this is not a great place to set the expiration time
		var expiration time.Duration = 30 * time.Duration(time.Minute)
		if expTimeMinutes, exists := os.LookupEnv("LLAMA_MODEL_TTL_MINUTES"); exists {
			ext, err := strconv.Atoi(expTimeMinutes)
			if err != nil {
				http.Error(w, "Invalid LLAMA_MODEL_TTL_MINUTES", http.StatusInternalServerError)
				return
			}
			expiration = time.Duration(ext) * time.Minute
		}
		cache.Set(req.Model, embdr, expiration)
	}
	embeddings, err := embdr.EmbedTexts(req.Texts)
	if err != nil {
		http.Error(w, "Failed to embed texts", http.StatusInternalServerError)
		return
	}

	resp := types.EmbedResponse{Embeddings: embeddings}
	w.Header().Set("Content-Type", "application/json")
	err = json.NewEncoder(w).Encode(resp)
	if err != nil {
		http.Error(w, "Failed to encode response", http.StatusInternalServerError)
		return
	}
}

func VersionHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"version": types.VERSION})
}

func HealthHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{"status": "running", "time": time.Now().Unix()})
}
