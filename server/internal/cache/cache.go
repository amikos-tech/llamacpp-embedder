package cache

import (
	"sync"
	"time"

	"github.com/amikos-tech/llamacpp-embedder/server/internal/worker"
)

type Cache struct {
	pools map[string]*worker.Pool
	mu    sync.RWMutex
}

func NewCache() *Cache {
	cache := &Cache{
		pools: make(map[string]*worker.Pool),
	}
	go cache.cleanupExpiredPools()
	return cache
}

func (c *Cache) cleanupExpiredPools() {
	ticker := time.NewTicker(1 * time.Minute)
	defer ticker.Stop()

	for range ticker.C {
		c.mu.Lock()
		for model, pool := range c.pools {
			if time.Since(pool.GetLastAccessed()) > 1*time.Minute {
				pool.Close()
				delete(c.pools, model)
			}
		}
		c.mu.Unlock()
	}
}

func (c *Cache) GetOrCreateWorkerPool(model string, workers int) (*worker.Pool, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if pool, found := c.pools[model]; found {
		return pool, nil
	}

	pool, err := worker.NewPool(model, workers)
	if err != nil {
		return nil, err
	}
	c.pools[model] = pool
	return pool, nil
}
