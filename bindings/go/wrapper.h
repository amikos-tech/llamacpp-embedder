//
// Created by Trayan Azarov on 13.09.24.
//
#include <stdint.h>
#include <stdbool.h>
#include <dlfcn.h>
#include <stdio.h>
#ifndef WRAPPER_H
#define WRAPPER_H

typedef struct {
    float *data;
    size_t rows;
    size_t cols;
} FloatMatrix;

#ifdef __cplusplus
extern "C" {
#endif

void *load_library(const char *shared_lib_path);
int init_llama_embedder(char *model_path, uint32_t pooling_type);
void free_llama_embedder();
FloatMatrix llama_embedder_embed(const char **texts, size_t text_count, int32_t norm);
void free_float_matrix(FloatMatrix fm);

const char* get_last_error();

char ** llama_embedder_get_metadata(size_t* size);
void free_metadata(char** metadata_array, size_t size);

#ifdef __cplusplus
}
#endif

#endif // WRAPPER_H