package cache

import (
	"fmt"
	"github.com/amikos-tech/llamacpp-embedder/server/internal/embedder"
	"github.com/robfig/cron/v3"
	"sync"
	"time"
)

type CacheItem struct {
	Value      *embedder.LlamaEmbedder
	Expiration int64 // time in minutes
}

type Cache struct {
	items map[string]CacheItem
	mu    sync.RWMutex
	chron *cron.Cron
}

func NewCache() (*Cache, error) {
	cache := &Cache{
		items: make(map[string]CacheItem),
		chron: cron.New(),
	}
	_, err := cache.chron.AddFunc("@every 30s", cache.EvictExpired)
	if err != nil {
		return nil, err
	}
	cache.chron.Start()
	return cache, nil
}

func (c *Cache) Set(key string, value *embedder.LlamaEmbedder, ttl time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.items[key] = CacheItem{
		Value:      value,
		Expiration: time.Now().Add(ttl).Unix() / 60, // store expiration time in minutes
	}
}

func (c *Cache) Get(key string) (*embedder.LlamaEmbedder, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	item, found := c.items[key]
	if !found || time.Now().Unix()/60 >= item.Expiration { // compare current time in minutes
		if found {
			delete(c.items, key)
		}
		return nil, false
	}
	return item.Value, true
}

func (c *Cache) Delete(key string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	delete(c.items, key)
}

func (c *Cache) Clear() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.items = make(map[string]CacheItem)
}

func (c *Cache) Len() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return len(c.items)
}

func (c *Cache) Keys() []string {
	c.mu.RLock()
	defer c.mu.RUnlock()
	keys := make([]string, 0, len(c.items))
	for k := range c.items {
		keys = append(keys, k)
	}
	return keys
}

func (c *Cache) EvictExpired() {
	c.mu.Lock()
	defer c.mu.Unlock()
	for key, item := range c.items {
		if time.Now().Unix()/60 >= item.Expiration { // compare current time in minutes
			fmt.Printf("Evicting expired embedder: %s\n", key)
			item.Value.Close()
			delete(c.items, key)
		}
	}
}
