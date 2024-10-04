package main

import (
	"fmt"
	"github.com/amikos-tech/llamacpp-embedder/server/internal/api"
	"github.com/amikos-tech/llamacpp-embedder/server/internal/middleware"
	"github.com/amikos-tech/llamacpp-embedder/server/internal/utils"
	"log"
	"net/http"
	"os"
	"runtime"
	"runtime/debug"
	"time"
)

func logMemoryUsage() {
	for {
		var m runtime.MemStats
		runtime.ReadMemStats(&m)
		log.Printf("Alloc = %v MiB", bToMb(m.Alloc))
		log.Printf("\tTotalAlloc = %v MiB", bToMb(m.TotalAlloc))
		log.Printf("\tSys = %v MiB", bToMb(m.Sys))
		log.Printf("\tNumGC = %v\n", m.NumGC)
		debug.FreeOSMemory()
		time.Sleep(1 * time.Minute)
	}
}

func bToMb(b uint64) uint64 {
	return b / 1024 / 1024
}

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
