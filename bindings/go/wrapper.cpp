//
// Created by Trayan Azarov on 13.09.24.
//
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <stdexcept>
#include "../../src/embedder.h"
#include "wrapper.h"

typedef llama_embedder * (*init_embedder_local_func)(const char *, uint32_t);
typedef void (*free_embedder_local_func)(llama_embedder *);
typedef void (*embed_local_func)(llama_embedder *, const std::vector<std::string> &, std::vector<std::vector<float>> &, int32_t);
typedef void (*get_metadata_local_func)(llama_embedder *, std::unordered_map<std::string, std::string> &);

void * libh = NULL;
llama_embedder * embedder = NULL;
init_embedder_local_func init_embedder_f = NULL;
free_embedder_local_func free_embedder_f = NULL;
embed_local_func embed_f = NULL;
get_metadata_local_func get_metadata_f = NULL;

static std::string last_error;

extern "C" {
const char* get_last_error() {
    return last_error.c_str();
}

void set_last_error(const char* error_message) {
    last_error = error_message;
}

void * load_library(const char * shared_lib_path){
    try {
        libh = dlopen(shared_lib_path, RTLD_LAZY);
        if (!libh) {
            std::string error_message = "Failed to load shared library: " + std::string(dlerror());
            throw std::runtime_error(error_message);
        }
        init_embedder_f = (llama_embedder *(*)(const char *, uint32_t)) dlsym(libh, "init_embedder");
        if (!init_embedder_f) {
            std::string error_message = "Failed to load init_embedder function: " + std::string(dlerror());
            throw std::runtime_error(error_message);
        }

        free_embedder_f = (void (*)(llama_embedder *)) dlsym(libh, "free_embedder");

        if (!free_embedder_f) {
            std::string error_message = "Failed to load free_embedder function: " + std::string(dlerror());
            throw std::runtime_error(error_message);
        }

        embed_f = (void (*)(llama_embedder *, const std::vector<std::string> &, std::vector<std::vector<float>> &, int32_t)) dlsym(libh, "embed");

        if (!embed_f) {
            std::string error_message = "Failed to load embed function: " + std::string(dlerror());
            throw std::runtime_error(error_message);
        }

        get_metadata_f = (void (*)(llama_embedder *, std::unordered_map<std::string, std::string> &)) dlsym(libh, "get_metadata");

        if (!get_metadata_f) {
            std::string error_message = "Failed to load get_metadata function: " + std::string(dlerror());
            throw std::runtime_error(error_message);
        }

        return libh;
    } catch (const std::exception &e) {
        std::string error_message = "Failed to load shared library: " + std::string(e.what());
        set_last_error(error_message.c_str());
        if (libh != NULL) {
            dlclose(libh);
        }
        return NULL;
    }
}

int init_llama_embedder(char * model_path, uint32_t pooling_type = 1 ) {
    if (!libh) {
        set_last_error("Shared library not loaded, use load_library first.");
        return -1;
    }
    try {
        embedder = init_embedder_f(model_path, 1);
        if (!embedder) {
            throw std::runtime_error("Embedder not initialized properly.");
        }
    } catch (const std::exception &e) {
        std::string error_message = "Failed to initialize embedder: " + std::string(e.what());
        set_last_error(error_message.c_str());
        return -1;
    }
    return 0;
}

void free_llama_embedder() {
    if (embedder) {
        free_embedder_f(embedder);
    }
    if (libh) {
        dlclose(libh);
    }
}

FloatMatrix llama_embedder_embed(const char** texts, size_t text_count, int32_t norm) {
    std::vector<std::string> texts_vec;
    for (size_t i = 0; i < text_count; i++) {
        texts_vec.push_back(texts[i]);
    }
    std::vector<std::vector<float>> output;
    embed_f(embedder, texts_vec, output, norm);

    FloatMatrix fm;
    fm.rows = output.size();
    fm.cols = output[0].size();
    fm.data = (float *) malloc(fm.rows * fm.cols * sizeof(float));
    for (size_t i = 0; i < fm.rows; i++) {
        for (size_t j = 0; j < fm.cols; j++) {
            fm.data[i * fm.cols + j] = output[i][j];
        }
    }
    return fm;
}

char ** llama_embedder_get_metadata(size_t* size){
    std::unordered_map<std::string, std::string> metadata;
    get_metadata_f(embedder, metadata);
    *size = metadata.size();
    char** metadata_array = (char**)malloc(metadata.size() * sizeof(char*));
    size_t i = 0;
    for (const auto& pair : metadata) {
        std::string entry = pair.first + "=" + pair.second;
        metadata_array[i] = (char*)malloc((entry.size() + 1) * sizeof(char));
        std::strcpy(metadata_array[i], entry.c_str());
        i++;
    }
    return metadata_array;
}

void free_metadata(char** metadata_array, size_t size) {
    for (size_t i = 0; i < size; i++) {
        free(metadata_array[i]);
    }
    free(metadata_array);
}

void free_float_matrix(FloatMatrix fm) {
    free(fm.data);
}


}
