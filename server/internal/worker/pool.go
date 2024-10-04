package worker

import (
	"fmt"
	"github.com/amikos-tech/llamacpp-embedder/server/internal/embedder"
	"github.com/amikos-tech/llamacpp-embedder/server/internal/types"
	"github.com/amikos-tech/llamacpp-embedder/server/internal/utils"
	"path/filepath"
	"sync"
	"time"
)

type Job struct {
	Request  *types.EmbedRequest
	Response chan *types.EmbedResponse
}

type Pool struct {
	jobs         chan Job
	workers      int
	model        string
	close        chan struct{}
	wg           sync.WaitGroup
	lastAccessed time.Time
	mu           sync.Mutex
}

func NewPool(model string, workers int) (*Pool, error) {

	pool := &Pool{
		jobs:    make(chan Job),
		workers: workers,
		model:   model,
		close:   make(chan struct{}),
	}
	err := pool.Start()
	if err != nil {
		return nil, err
	}
	return pool, nil
}

func (p *Pool) Start() (err error) {
	for i := 0; i < p.workers; i++ {
		p.wg.Add(1)
		go func() {
			defer p.wg.Done()
			err = p.worker()
			if err != nil {
				fmt.Printf("worker error: %v", err)
				return
			}
		}()
		if err != nil {
			return
		}
	}
	return nil
}

func (p *Pool) worker() error {
	emb, closeEmbedder, err := embedder.NewLlamaEmbedder(filepath.Join(utils.GetModelCacheDir(), p.model))
	if err != nil {
		return fmt.Errorf("failed to create embedder: %v", err)
	}
	defer closeEmbedder()
	for {
		select {
		case job := <-p.jobs:
			p.updateLastAccessed()
			embeddings, err := emb.EmbedTexts(job.Request.Texts)
			if err != nil {
				job.Response <- &types.EmbedResponse{Error: err.Error()}
			} else {
				job.Response <- &types.EmbedResponse{Embeddings: embeddings}
			}
		case <-p.close:
			return nil
		}
	}
}

func (p *Pool) updateLastAccessed() {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.lastAccessed = time.Now()
}

func (p *Pool) GetLastAccessed() time.Time {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.lastAccessed
}

func (p *Pool) Submit(job Job) {
	p.updateLastAccessed()
	p.jobs <- job
}

func (p *Pool) Close() {
	close(p.close)
	p.wg.Wait()
	close(p.jobs)
}
