//
// Created by Trayan Azarov on 28.08.24.
//
#include <vector>
#include <unordered_map>

#ifndef LLAMA_CPP_EMBEDDING_H
#define LLAMA_CPP_EMBEDDING_H
#endif //LLAMA_CPP_EMBEDDING_H

#pragma once

#if defined(_WIN32) || defined(_WIN64)
#if defined(BUILDING_DLL)
        #define EXPORT_SYMBOL __declspec(dllexport)
    #else
        #define EXPORT_SYMBOL __declspec(dllimport)
    #endif
#else
#define EXPORT_SYMBOL __attribute__((visibility("default")))
#endif


struct llama_embedder {
    struct llama_model   * model   = nullptr;
    struct llama_context * context = nullptr;
    std::unordered_map<std::string, std::string> model_metadata;
};

struct llama_tokenizer_data {
    std::vector<int32_t> tokens;
    std::vector<int32_t> attention_mask;
};

extern "C" {
typedef struct {
    float *data;
    size_t rows;
    size_t cols;
} FloatMatrix;

typedef struct {
    const char* key;
    const char* value;
} MetadataPair;

EXPORT_SYMBOL llama_embedder * init_embedder(const char * embedding_model, uint32_t pooling_type) noexcept(false);
EXPORT_SYMBOL void free_embedder(llama_embedder *embedder) noexcept;
EXPORT_SYMBOL void embed(llama_embedder * embedder, const std::vector<std::string> & texts, std::vector<std::vector<float>> & output, int32_t embd_norm) noexcept(false);
EXPORT_SYMBOL FloatMatrix embed_c(llama_embedder * embedder, const char  ** texts,size_t  text_len, int32_t embd_norm) noexcept(false);
EXPORT_SYMBOL void free_float_matrix(FloatMatrix * floatMatrix);
EXPORT_SYMBOL void get_metadata(llama_embedder * embedder, std::unordered_map<std::string, std::string> &output) noexcept(false);
EXPORT_SYMBOL int get_metadata_c(llama_embedder * embedder,MetadataPair** pairs, size_t* count) noexcept(false);
EXPORT_SYMBOL void free_metadata_c(MetadataPair* metadata_array, size_t size);
EXPORT_SYMBOL void tokenize(llama_embedder * embedder, const std::vector<std::string>& texts, std::vector<llama_tokenizer_data> &output, bool add_special_tokens = true, bool parse_special = false, bool enable_padding = false) noexcept(false);
}