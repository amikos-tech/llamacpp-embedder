//
// Created by Trayan Azarov on 13.09.24.
//
#ifndef WRAPPER_H
#define WRAPPER_H
#if defined(_WIN32) || defined(_WIN64)
#define EXPORT_GO_WRAPPER __declspec(dllexport)
#include <windows.h>
    typedef HMODULE lib_handle;
#else
#define EXPORT_GO_WRAPPER __attribute__((visibility("default")))
#include <dlfcn.h>
    typedef void* lib_handle;
#endif
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>



#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float *data;
    size_t rows;
    size_t cols;
} FloatMatrix;

typedef struct {
    const char* key;
    const char* value;
} MetadataPair;

EXPORT_GO_WRAPPER lib_handle load_library(const char *shared_lib_path);
EXPORT_GO_WRAPPER int init_llama_embedder(char *model_path, uint32_t pooling_type);
EXPORT_GO_WRAPPER void free_llama_embedder();
EXPORT_GO_WRAPPER FloatMatrix llama_embedder_embed(const char **texts, size_t text_count, int32_t norm);
EXPORT_GO_WRAPPER void free_float_matrixw(FloatMatrix * fm);

EXPORT_GO_WRAPPER const char* get_last_error();

EXPORT_GO_WRAPPER char ** llama_embedder_get_metadata(size_t* size);
EXPORT_GO_WRAPPER void free_metadata(char** metadata_array, size_t size);

#ifdef __cplusplus
}
#endif

#endif // WRAPPER_H