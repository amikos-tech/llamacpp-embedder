const { LlamaEmbedder, PoolingType, NormalizationType } = require('./llama_embedder');
async function runTest() {
    try {
        // Initialize the LlamaEmbedder
        const embedder = new LlamaEmbedder('/Users/tazarov/Downloads/all-MiniLM-L6-v2.Q4_0.gguf', {
            poolingType: PoolingType.MEAN,
            normalizationType: NormalizationType.EUCLIDEAN
        });

        // Test the embed method with sample text
        const texts = ["Hello, world!", "Testing embeddings"];
        const embeddings = embedder.embed(texts);
        console.log("Embeddings:", embeddings);

        // Test metadata retrieval
        const metadata = embedder.getMetadata();
        console.log("Metadata:", metadata);
    } catch (error) {
        console.error("Error during test:", error);
    }
}
const defaultHFRepo = "leliuga/all-MiniLM-L6-v2-GGUF"
const defaultModelFile = "all-MiniLM-L6-v2.Q4_0.gguf"
async function runTestHfModel() {
    try {
        // Initialize the LlamaEmbedder
        const embedder = new LlamaEmbedder(defaultModelFile, {
            hfRepository: defaultHFRepo,
            poolingType: PoolingType.MEAN,
            normalizationType: NormalizationType.EUCLIDEAN
        });

        // Test the embed method with sample text
        const texts = ["Hello, world!", "Testing embeddings"];
        const embeddings = await embedder.embed(texts);
        console.log("Embeddings:", embeddings);

        // Test metadata retrieval
        const metadata = await embedder.getMetadata();
        console.log("Metadata:", metadata);
    } catch (error) {
        console.error("Error during test:", error);
    }
}

runTestHfModel();