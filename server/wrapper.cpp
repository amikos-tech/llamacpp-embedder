#include "wrapper.h"
#include "../src/embedder.h"
#include <iostream>
#include <string>
#include <vector>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <stdint.h>
#include <mutex>
#include "atomic"
#include <thread>

static std::string last_error;
static std::mutex embedder_mutex;
extern "C" {

const char* get_last_error() {
    std::lock_guard<std::mutex> lock(embedder_mutex);
    return last_error.c_str();
}

void set_last_error(const char* error_message) {
    std::lock_guard<std::mutex> lock(embedder_mutex);
    last_error = error_message;
}

int init_embedder_l(llama_embedder** out_embedder, const char* model_path, uint32_t pooling_type) {
    try {
        *out_embedder = init_embedder(model_path, pooling_type);
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "Error: %s\n", e.what());
        return -1;
    }
}

void free_embedder_l(llama_embedder *embedder) {
    free_embedder(embedder);
}

FloatMatrixW embed_texts(llama_embedder *embedder, const char ** texts, size_t text_count, int32_t norm){
        try {
            std::vector<std::vector<float>> output;
            FloatMatrix fm = embed_c(embedder, texts, text_count, norm);
            FloatMatrixW fmw = {fm.data, fm.rows, fm.cols};
            return fmw;
        } catch (const std::exception &e) {
            set_last_error(e.what());
        }
        return {nullptr, 0, 0};
}
void free_float_matrixw(FloatMatrixW * fm) {
    if (fm != nullptr){
        if (fm->data != nullptr) {
            free(fm->data);
            fm->data = nullptr;
        }
    }
}
}