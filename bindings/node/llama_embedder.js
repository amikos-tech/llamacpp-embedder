const fs = require('fs');
const path = require('path');
const {pipeline} = require('stream/promises');

// Enums to mirror Python version
const PoolingType = {
    NONE: 0,
    MEAN: 1,
    CLS: 2,
    LAST: 3,
};

const NormalizationType = {
    NONE: -1,
    MAX_ABS_INT16: 0,
    TAXICAB: 1,
    EUCLIDEAN: 2,
    // Add other normalization types as needed
};

class LlamaEmbedder {
    constructor(modelPath, {
        poolingType = PoolingType.MEAN,
        normalizationType = NormalizationType.EUCLIDEAN,
        hfRepository = null,
        hfToken = process.env.HF_TOKEN
    } = {}) {
        this.modelPath = modelPath;
        this.poolingType = poolingType;
        this.normalizationType = normalizationType;
        this.hfRepository = hfRepository;
        this.hfToken = hfToken;
        this._hfInitialized = false;

        if (!this.hfRepository) {
            this.initializeFromLocalPath();
            this._hfInitialized = true;
        }
    }

    async initializeFromHuggingFace() {
        const modelUrl = `https://huggingface.co/${this.hfRepository}/resolve/main/${this.modelPath}`;
        const outputDir = path.join(process.cwd(), 'models');
        const outputPath = path.join(outputDir, path.basename(this.modelPath));

        if (!fs.existsSync(outputPath)) {
            console.log(`Downloading model from ${modelUrl}...`);
            await this.downloadModel(modelUrl, outputPath);
            console.log(`Model downloaded to ${outputPath}`);
        } else {
            console.log(`Model already exists at ${outputPath}`);
        }

        this.initializeNativeEmbedder(outputPath);
    }

    initializeFromLocalPath() {
        this.initializeNativeEmbedder(this.modelPath);
    }

    initializeNativeEmbedder(modelPath) {
        // const { LlamaEmbedder: NativeLlamaEmbedder } = require('bindings')('llama_embedder');
        const nativeBinding = require('node-gyp-build')(path.join(__dirname));
        const {LlamaEmbedder: NativeLlamaEmbedder} = nativeBinding;
        this.embedder = new NativeLlamaEmbedder(modelPath, this.poolingType, this.normalizationType);
    }

    async downloadModel(modelUrl, outputPath) {
        const response = await fetch(modelUrl, {
            headers: this.hfToken ? {'Authorization': `Bearer ${this.hfToken}`} : {}
        });
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        await fs.promises.mkdir(path.dirname(outputPath), {recursive: true});
        const fileStream = fs.createWriteStream(outputPath);
        await pipeline(response.body, fileStream);
    }

    async embed(texts) {
        if (this._hfInitialized) {
            return this.embedder.embed(texts, this.normalizationType);
        } else {
            return await this.initializeFromHuggingFace().then(() => {
                const res = this.embedder.embed(texts, this.normalizationType);
                this._hfInitialized = true;
                return res;

            }).catch((error) => {
                console.error("Error during embedding:", error);
            });
        }
    }

    async getMetadata() {

        if (this._hfInitialized) {
            return this.embedder.getMetadata();
        } else {
            return await this.initializeFromHuggingFace().then(() => {
                const res = this.embedder.getMetadata();
                this._hfInitialized = true;
                return res;

            }).catch((error) => {
                console.error("Error during embedding:", error);
            });
        }
    }
}

module.exports = {LlamaEmbedder, PoolingType, NormalizationType};