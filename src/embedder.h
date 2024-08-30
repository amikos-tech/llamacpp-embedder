//
// Created by Trayan Azarov on 28.08.24.
//
#include <vector>

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
};


extern "C" {
EXPORT_SYMBOL llama_embedder * init_embedder(const char * embedding_model, const uint32_t pooling_type);
EXPORT_SYMBOL void free_embedder(llama_embedder *embedder);
EXPORT_SYMBOL void embed(llama_embedder * embedder, const std::vector<std::string> prompts, std::vector<std::vector<float>> &output, int32_t embd_norm);
}