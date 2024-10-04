package main

import (
	"fmt"
	"github.com/amikos-tech/llamacpp-embedder/server/internal/api"
	"github.com/amikos-tech/llamacpp-embedder/server/internal/middleware"
	"github.com/amikos-tech/llamacpp-embedder/server/internal/utils"
	"log"
	"net/http"
	"os"
)

func main() {
	err := utils.EnsureCacheDir()
	if err != nil {
		panic(err)
	}
	if modelsToDownload, exists := os.LookupEnv("LLAMA_CACHED_MODELS"); exists {
		err := utils.EnsureModels(modelsToDownload)
		if err != nil {
			panic(err)
		}
	}
	mux := http.NewServeMux()
	mux.Handle("GET /embed_models", middleware.LoggingMiddleware(middleware.CachingMiddleware(http.HandlerFunc(api.EmbedModelsHandler))))
	mux.Handle("POST /embed_texts", middleware.LoggingMiddleware(middleware.CachingMiddleware(http.HandlerFunc(api.EmbedTextsHandler))))
	mux.Handle("GET /version", middleware.LoggingMiddleware(http.HandlerFunc(api.VersionHandler)))
	mux.Handle("GET /health", middleware.LoggingMiddleware(http.HandlerFunc(api.HealthHandler)))

	var port = fmt.Sprintf(":%d", 8080)
	if envPort, exists := os.LookupEnv("PORT"); exists {
		port = fmt.Sprintf(":%s", envPort)
	}
	log.Printf("Server starting on port %s", port)
	err = http.ListenAndServe(port, mux)
	if err != nil {
		log.Fatalf("Server failed to start: %v", err)
	}
}
