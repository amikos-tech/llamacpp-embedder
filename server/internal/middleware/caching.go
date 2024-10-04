package middleware

import (
	"context"
	cache2 "github.com/amikos-tech/llamacpp-embedder/server/internal/cache"
	"net/http"
)

var cache *cache2.Cache

func init() {
	cache = cache2.NewCache()
}

func CachingMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		ctx := context.WithValue(r.Context(), "cache", cache)
		next.ServeHTTP(w, r.WithContext(ctx))
	})
}
