
#ifndef WRAPPER_H
#define WRAPPER_H
#if defined(_WIN32) || defined(_WIN64)
#define EXPORT_GO_WRAPPER __declspec(dllexport)
#else
#define EXPORT_GO_WRAPPER __attribute__((visibility("default")))
#endif
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct llama_embedder llama_embedder;

typedef struct {
    float *data;
    size_t rows;
    size_t cols;
} FloatMatrixW;

EXPORT_GO_WRAPPER int init_embedder_l(llama_embedder**, const char*, uint32_t);
EXPORT_GO_WRAPPER void free_embedder_l(llama_embedder *embedder);
EXPORT_GO_WRAPPER FloatMatrixW embed_texts(llama_embedder *, const char **, size_t, int32_t);
EXPORT_GO_WRAPPER void free_float_matrixw(FloatMatrixW * fm);
EXPORT_GO_WRAPPER const char* get_last_error();
#ifdef __cplusplus
}
#endif

#endif // WRAPPER_H