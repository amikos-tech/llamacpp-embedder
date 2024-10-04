#include "common.h"
#include "llama.h"
#include "embedder.h"
#include <ctime>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

static std::vector<std::string> split_lines(const std::string &s, const std::string &separator = "\n") {
    std::vector<std::string> lines;
    size_t start = 0;
    size_t end = s.find(separator);

    while (end != std::string::npos) {
        lines.push_back(s.substr(start, end - start));
        start = end + separator.length();
        end = s.find(separator, start);
    }

    lines.push_back(s.substr(start)); // Add the last part

    return lines;
}

static void batch_add_seq(llama_batch &batch, const std::vector<int32_t> &tokens, llama_seq_id seq_id) {
    size_t n_tokens = tokens.size();
    for (size_t i = 0; i < n_tokens; i++) {
        llama_batch_add(batch, tokens[i], i, {seq_id}, true);
    }
}

static void batch_decode(llama_context *ctx, llama_batch &batch, float *output, int n_seq, int n_embd, int embd_norm) {
    const enum llama_pooling_type pooling_type = llama_pooling_type(ctx);
    const struct llama_model *model = llama_get_model(ctx);

    // clear previous kv_cache values (irrelevant for embeddings)
    llama_kv_cache_clear(ctx);

    // run model
    if (llama_model_has_encoder(model) && !llama_model_has_decoder(model)) {
        // encoder-only model - BERT-like models.
        if (llama_encode(ctx, batch) < 0) {
            fprintf(stderr, "%s : failed to encode\n", __func__);
        }
    } else if (!llama_model_has_encoder(model) && llama_model_has_decoder(model)) {
        // decoder-only model
        if (llama_decode(ctx, batch) < 0) {
            fprintf(stderr, "%s : failed to decode\n", __func__);
        }
    }

    for (int i = 0; i < batch.n_tokens; i++) {
        if (!batch.logits[i]) {
            continue;
        }

        const float *embd = nullptr;
        int embd_pos = 0;

        if (pooling_type == LLAMA_POOLING_TYPE_NONE) {
            // try to get token embeddings
            embd = llama_get_embeddings_ith(ctx, i);
            embd_pos = i;
            GGML_ASSERT(embd != NULL && "failed to get token embeddings");
        } else {
            // try to get sequence embeddings - supported only when pooling_type is not NONE
            embd = llama_get_embeddings_seq(ctx, batch.seq_id[i][0]);
            embd_pos = batch.seq_id[i][0];
            GGML_ASSERT(embd != NULL && "failed to get sequence embeddings");
        }

        float *out = output + embd_pos * n_embd;
        llama_embd_normalize(embd, out, n_embd, embd_norm);
    }
}

void my_log_callback(enum ggml_log_level level, const char *text, void *user_data) {
    // Do nothing, effectively silencing the log
}

// Function to generate attention mask
std::vector<int32_t> generate_attention_mask(const std::vector<int>& token_ids, unsigned long  max_length) {
    std::vector<int32_t> attention_mask(max_length, 0);  // Initialize mask with 0s

    for (size_t i = 0; i < token_ids.size() && i < max_length; ++i) {
        if (token_ids[i] != 0) {
            attention_mask[i] = 1;  // Set 1 for non-padding tokens (non-zero)
        }
    }

    return attention_mask;
}

/// Function to pad token IDs and add CLS and SEP tokens
std::vector<int> pad_tokens(const std::vector<int>& token_ids, unsigned long max_length,
                                            int pad_token_id = 0) {
    std::vector<int> padded_token_ids;

    // Add the actual tokens
    padded_token_ids.insert(padded_token_ids.end(), token_ids.begin(), token_ids.end());

    // Add padding if token size is still less than max_length
    if (padded_token_ids.size() < max_length) {
        padded_token_ids.resize(max_length, pad_token_id);
    }

    return padded_token_ids;
}

enum llama_pooling_type from_uint(const uint32_t pooling_type){
    switch (pooling_type) {
        case 0:
            return LLAMA_POOLING_TYPE_NONE;
        case 1:
            return LLAMA_POOLING_TYPE_MEAN;
        case 2:
            return LLAMA_POOLING_TYPE_CLS;
        case 3:
            return LLAMA_POOLING_TYPE_LAST;
        default:
            throw std::runtime_error("error: invalid pooling type");
    }
}

llama_embedder *init_embedder(const char *embedding_model, const uint32_t pooling_type) {
    gpt_params params;

    log_disable();

    params.model = embedding_model;
    params.embedding = true;
    // For non-causal models, batch size must be equal to ubatch size
    params.n_ubatch = params.n_batch;
    params.pooling_type = from_uint(pooling_type);


    if (params.seed == LLAMA_DEFAULT_SEED) {
        params.seed = time(nullptr);
    }


    std::mt19937 rng(params.seed);

    llama_backend_init();
    llama_numa_init(params.numa);


    llama_log_set(my_log_callback, nullptr);
    // load the model
    llama_init_result llama_init = llama_init_from_gpt_params(params);

    llama_model *model = llama_init.model;
    llama_context *ctx = llama_init.context;
    if (model == nullptr) {
        fprintf(stderr, "%s: error: unable to load model\n", __func__);
        throw std::runtime_error("error: unable to load model");
    }

    const int32_t n_ctx_train = llama_n_ctx_train(model);
    const uint32_t n_ctx = llama_n_ctx(ctx);

    if (llama_model_has_encoder(model) && llama_model_has_decoder(model)) {
        throw std::runtime_error("error: computing embeddings in encoder-decoder models is not supported");
    }

    if (n_ctx > n_ctx_train) {
        fprintf(stderr, "%s: warning: model was trained on only %d context tokens (%d specified)\n",
                __func__, n_ctx_train, n_ctx);
    }
    std::unordered_map<std::string, std::string> model_metadata;
    for (int i = 0; i < llama_model_meta_count(model); ++i) {
        char key[256];         // Buffer to hold the string result
        size_t key_size = sizeof(key);
        char value[1024];
        size_t value_size = sizeof(value);
        llama_model_meta_key_by_index(model, i,key,key_size);
        llama_model_meta_val_str_by_index(model, i,value,value_size);
        model_metadata[key] = value;
    }

    auto *embedder = new llama_embedder;
    embedder->context = ctx;
    embedder->model = model;
    embedder->model_metadata = model_metadata;
    return embedder;
}

void tokenize(llama_embedder *embedder, const std::vector<std::string>& texts, std::vector<llama_tokenizer_data> &output,const bool add_special_tokens, const bool parse_special, const bool enable_padding) {
    if (!embedder) {
        throw std::runtime_error("Error: Null pointer passed to tokenize function");
    }
    if (texts.empty()){
        fprintf(stderr, "Warn: empty texts.\n");
        return;
    }

    char model_arch[1024];
    size_t vmodel_arch_size = sizeof(model_arch);
    llama_model_meta_val_str(embedder->model, "general.architecture",model_arch, vmodel_arch_size);
    if (strcmp(model_arch, "bert") != 0) {
        throw std::runtime_error("error: tokenize function is only supported for BERT-like models");
    }

    for (const auto &text: texts) {
        auto tokens = ::llama_tokenize(embedder->context, text, add_special_tokens, parse_special);
        char value[1024];
        size_t value_size = sizeof(value);
        llama_model_meta_val_str(embedder->model, "bert.context_length",value, value_size);
        unsigned long max_length = tokens.size();
        if (enable_padding) {
            max_length = std::stoi(value);
            memset(value, 0, value_size);
            llama_model_meta_val_str(embedder->model, "tokenizer.ggml.padding_token_id",value, value_size);
            int padding_token_id = std::stoi(value);
            tokens = pad_tokens(tokens,max_length , padding_token_id);
        }
        auto attention_mask = generate_attention_mask(tokens, max_length);
        output.push_back({tokens, attention_mask});
    }
}


int get_metadata_c(llama_embedder * embedder, MetadataPair** pairs, size_t* count){
    std::unordered_map<std::string, std::string> metadata;
    get_metadata(embedder, metadata);
    *count = metadata.size();
    *pairs = (MetadataPair*)malloc(metadata.size() * sizeof(MetadataPair));
    size_t i = 0;
    for (const auto &pair : metadata) {
        char* key_copy = strdup(pair.first.c_str());
        char* value_copy = strdup(pair.second.c_str());

        if (key_copy == nullptr || value_copy == nullptr) {
            fprintf(stderr, "Failed to allocate memory for key or value at index %zu\n", i);
            // Handle allocation failure
            continue;
        }

        (*pairs)[i].key = key_copy;
        (*pairs)[i].value = value_copy;

        i++;
    }
    return 0;
}

void free_metadata_c(MetadataPair* metadata_array, size_t size) {
    for (size_t i = 0; i < size; i++) {
        if (metadata_array[i].key != nullptr) {
            free((void *) metadata_array[i].key);
        }
        if (metadata_array[i].value != nullptr) {
            free((void *) metadata_array[i].value);
        }
    }
    free(metadata_array);
}

void get_metadata(llama_embedder *embedder, std::unordered_map<std::string, std::string> &output) {
    output = embedder->model_metadata;
}


void free_embedder(llama_embedder *embedder) noexcept {
    if (!embedder) {
        return;
    }
    if (embedder->model) {
        llama_free_model(embedder->model);
    }
    if (embedder->context) {
        llama_free(embedder->context);
    }
    llama_backend_free();
    delete embedder;
}

FloatMatrix embed_c(llama_embedder * embedder, const char  ** texts,size_t  text_len, int32_t embd_norm){
    std::vector<std::string> texts_inner;
    texts_inner.reserve(text_len);
    for (size_t i = 0; i < text_len; i++) {
        texts_inner.emplace_back(texts[i]);
    }
    std::vector<std::vector<float>> output;
    FloatMatrix floatMatrix;
    try{
        embed(embedder, texts_inner, output, embd_norm);
    } catch (const std::exception &e) {
        fprintf(stderr, "Error: %s\n", e.what());
        floatMatrix.rows = 0;
        floatMatrix.cols = 0;
        floatMatrix.data = nullptr;
        return floatMatrix;
    }
    if (output.empty()) {
        floatMatrix.rows = 0;
        floatMatrix.cols = 0;
        floatMatrix.data = nullptr;
        return floatMatrix;
    }
    floatMatrix.rows = output.size();
    floatMatrix.cols = output[0].size();
    floatMatrix.data = (float *)malloc(floatMatrix.rows * floatMatrix.cols * sizeof(float));
    for (size_t i = 0; i < floatMatrix.rows; i++) {
        for (size_t j = 0; j < floatMatrix.cols; j++) {
            floatMatrix.data[i * floatMatrix.cols + j] = output[i][j];
        }
    }
    return floatMatrix;
}

void free_float_matrix(FloatMatrix * floatMatrix) {
    if (floatMatrix!= nullptr) {
        if (floatMatrix->data != nullptr) {
            free(floatMatrix->data);
            floatMatrix->data = nullptr;
        }
    }

}

// Creates embeddings from list of strings
void embed(llama_embedder *embedder, const std::vector<std::string> & texts, std::vector<std::vector<float>> & output,
           int32_t embd_norm) {
    if (!embedder) {
        throw std::runtime_error("Error: Null pointer passed to embed function");
    }
    if (texts.empty()){
        fprintf(stderr, "Warn: empty prompts.\n");
        return;
    }
    if (!output.empty()){
        fprintf(stderr, "Warn: output is not empty.\n");
        return;
    }
    llama_context *ctx = embedder->context;
    llama_model *model = embedder->model;
    const enum llama_pooling_type pooling_type = llama_pooling_type(ctx);


    // max batch size
    const uint32_t n_batch = llama_n_batch(ctx);//params.n_batch;
    GGML_ASSERT(llama_n_batch(ctx) >= llama_n_ctx(ctx));

    // tokenize the prompts and trim
    std::vector<std::vector<int32_t>> inputs;
    std::vector<llama_tokenizer_data> output_token_data;
    ::tokenize(embedder, texts, output_token_data);
    for (const auto &tokenizer_data : output_token_data) {
        auto inp = tokenizer_data.tokens;
        if (inp.size() > n_batch) {
            fprintf(stderr,
                    "%s: error: number of tokens in input line (%lld) exceeds batch size (%lld), increase batch size and re-run\n",
                    __func__, (long long int) inp.size(), (long long int) n_batch);
            throw std::runtime_error("error: number of tokens in input line exceeds batch size");
        }
        inputs.push_back(inp);
    }

    // check if the last token is SEP
    // it should be automatically added by the tokenizer when 'tokenizer.ggml.add_eos_token' is set to 'true'
    for (auto &inp: inputs) {
        if (inp.empty() || inp.back() != llama_token_sep(model)) {
            fprintf(stderr, "%s: warning: last token in the prompt is not SEP\n", __func__);
            fprintf(stderr, "%s:          'tokenizer.ggml.add_eos_token' should be set to 'true' in the GGUF header\n",
                    __func__);
        }
    }

    // initialize batch
    const size_t n_prompts = texts.size();
    struct llama_batch batch = llama_batch_init( (int32_t )n_batch, 0, 1);

    // count number of embeddings
    size_t n_embd_count = 0;
    if (pooling_type == LLAMA_POOLING_TYPE_NONE) {
        for (int k = 0; k < n_prompts; k++) {
            n_embd_count += inputs[k].size();
        }
    } else {
        n_embd_count = n_prompts;
    }

    // allocate output
    const int n_embd = llama_n_embd(model);
    std::vector<float> embeddings(n_embd_count * n_embd, 0);
    float *emb = embeddings.data();
    // Resize the outer vector to have n_prompts rows
    output.resize(n_prompts);

    // Resize each inner vector to have n_embd columns
    for (int i = 0; i < n_prompts; ++i) {
        output[i].resize(n_embd);
    }

    // break into batches
    int e = 0; // number of embeddings already stored
    int s = 0; // number of prompts in current batch
    for (int k = 0; k < n_prompts; k++) {
        // clamp to n_batch tokens
        auto &inp = inputs[k];

        const uint64_t n_toks = inp.size();

        // encode if at capacity
        if (batch.n_tokens + n_toks > n_batch) {
            float *out = emb + e * n_embd;
            batch_decode(ctx, batch, out, s, n_embd, embd_norm);
            e += pooling_type == LLAMA_POOLING_TYPE_NONE ? batch.n_tokens : s;
            s = 0;
            llama_batch_clear(batch);
        }

        // add to batch
        batch_add_seq(batch, inp, s);
        s += 1;
    }

    // final batch
    float *out = emb + e * n_embd;
    batch_decode(ctx, batch, out, s, n_embd, embd_norm);


    if (pooling_type == LLAMA_POOLING_TYPE_NONE) {
        for (int j = 0; j < n_embd_count; j++) {
            for (int i = 0; i < n_embd; i++) {
                output[j][i] = emb[j * n_embd + i];
            }
        }
    } else {
        for (int j = 0; j < n_prompts; j++) {
            for (int i = 0; i < n_embd; i++) {
                output[j][i] = emb[j * n_embd + i];
            }
        }
    }
    llama_batch_free(batch);
}