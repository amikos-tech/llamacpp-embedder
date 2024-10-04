package types

const VERSION = "1.0.0"

type EmbedModelListResponse struct {
	Models []string `json:"models"`
	Error  string   `json:"error"`
}

type EmbedRequest struct {
	Model string   `json:"model"`
	Texts []string `json:"texts"`
}

type EmbedResponse struct {
	Embeddings [][]float32 `json:"embeddings"`
	Error      string      `json:"error"`
}
