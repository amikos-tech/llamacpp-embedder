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
        // encoder-only model
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


    auto *embedder = new llama_embedder;
    embedder->context = ctx;
    embedder->model = model;
    return embedder;
}


void free_embedder(llama_embedder *embedder) {
    if (embedder->model) {
        llama_free_model(embedder->model);
    }
    if (embedder->context) {
        llama_free(embedder->context);
    }
    llama_backend_free();
    delete embedder;
}

// Creates embeddings from list of strings
void embed(llama_embedder *embedder, const std::vector<std::string> prompts, std::vector<std::vector<float>> &output,
           int32_t embd_norm) {
    if (!embedder) {
        throw std::runtime_error("Error: Null pointer passed to embed function");
    }
    if (prompts.empty()){
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
    const uint64_t n_batch = llama_n_batch(ctx);//params.n_batch;
    GGML_ASSERT(llama_n_batch(ctx) >= llama_n_ctx(ctx));

    // tokenize the prompts and trim
    std::vector<std::vector<int32_t>> inputs;
    for (const auto &prompt: prompts) {
        auto inp = ::llama_tokenize(ctx, prompt, true, false);
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
    const int n_prompts = prompts.size();
    struct llama_batch batch = llama_batch_init((long long int) n_batch, 0, 1);

    // count number of embeddings
    int n_embd_count = 0;
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
}