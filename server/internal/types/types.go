package types

const VERSION = "1.0.0"

type EmbedModelList struct {
	Models []string `json:"models"`
}

type EmbedRequest struct {
	Model string   `json:"model"`
	Texts []string `json:"texts"`
}

type EmbedResponse struct {
	Embeddings [][]float32 `json:"embeddings"`
}
