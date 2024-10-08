//
// Created by Trayan Azarov on 13.09.24.
//
#include <string>
#include <vector>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <stdexcept>
#include "wrapper.h"
#include <mutex>
#include "atomic"
#include <thread>

static std::mutex embedder_mutex;
typedef struct llama_embedder llama_embedder;

#if defined(_WIN32) || defined(_WIN64)

// Helper function to get the last error message on Windows
std::string GetLastErrorAsString() {
    DWORD errorMessageID = ::GetLastError();
    if(errorMessageID == 0) {
        return std::string(); // No error message has been recorded
    }

    LPSTR messageBuffer = nullptr;
    size_t size = FormatMessageA(
        FORMAT_MESSAGE_ALLOCATE_BUFFER |
        FORMAT_MESSAGE_FROM_SYSTEM |
        FORMAT_MESSAGE_IGNORE_INSERTS,
        nullptr,
        errorMessageID,
        MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        (LPSTR)&messageBuffer,
        0,
        nullptr
    );

    std::string message(messageBuffer, size);
    LocalFree(messageBuffer);
    return message;
}
#endif

#if defined(_WIN32) || defined(_WIN64)
#if defined(_WIN64)
        typedef llama_embedder* (*init_embedder_local_func)(const char*, uint32_t);
        typedef void (*free_embedder_local_func)(llama_embedder*);
        typedef FloatMatrix (*embed_c_local_func)(llama_embedder*, const char**, size_t, int32_t);
        typedef int (*get_metadata_c_local_func)(llama_embedder*, MetadataPair**, size_t*);
        typedef void (*free_metadata_c_local_func)(MetadataPair*, size_t);
    #else
        typedef llama_embedder* (__cdecl *init_embedder_local_func)(const char*, uint32_t);
        typedef void (__cdecl *free_embedder_local_func)(llama_embedder*);
        typedef FloatMatrix (__cdecl *embed_c_local_func)(llama_embedder*, const char**, size_t, int32_t);
        typedef int (__cdecl *get_metadata_c_local_func)(llama_embedder*, MetadataPair**, size_t*);
        typedef void (__cdecl *free_metadata_c_local_func)(MetadataPair*, size_t);
    #endif
#else
    typedef llama_embedder* (*init_embedder_local_func)(const char*, uint32_t);
    typedef void (*free_embedder_local_func)(llama_embedder*);
    typedef FloatMatrix (*embed_c_local_func)(llama_embedder*, const char**, size_t, int32_t);
    typedef int (*get_metadata_c_local_func)(llama_embedder*, MetadataPair**, size_t*);
    typedef void (*free_metadata_c_local_func)(MetadataPair*, size_t);
#endif

std::atomic<int> library_ref_count(0);
lib_handle libh = nullptr;
llama_embedder * embedder = nullptr;
init_embedder_local_func init_embedder_f = nullptr;
free_embedder_local_func free_embedder_f = nullptr;
embed_c_local_func embed_f = nullptr;
get_metadata_c_local_func get_metadata_f = nullptr;
free_metadata_c_local_func free_metadata_f = nullptr;

static std::string last_error;

extern "C" {
const char* get_last_error() {
    std::lock_guard<std::mutex> lock(embedder_mutex);
    return last_error.c_str();
}

void set_last_error(const char* error_message) {
    std::lock_guard<std::mutex> lock(embedder_mutex);
    last_error = error_message;
}

lib_handle load_library(const char * shared_lib_path){
    std::lock_guard<std::mutex> lock(embedder_mutex);
    try {
#if defined(_WIN32) || defined(_WIN64)
        libh = LoadLibraryA(shared_lib_path);
        if (!libh) {
            std::string error_message = "Failed to load shared library: " + GetLastErrorAsString();
            throw std::runtime_error(error_message);
        }
        init_embedder_f = reinterpret_cast<init_embedder_local_func>(GetProcAddress(libh, "init_embedder"));
        if (!init_embedder_f) {
            std::string error_message = "Failed to load init_embedder function: " + GetLastErrorAsString();
            throw std::runtime_error(error_message);
        }
        free_embedder_f = reinterpret_cast<free_embedder_local_func>(GetProcAddress(libh, "free_embedder"));
        if (!free_embedder_f) {
            std::string error_message = "Failed to load free_embedder function: " + GetLastErrorAsString();
            throw std::runtime_error(error_message);
        }
        embed_f = reinterpret_cast<embed_c_local_func>(GetProcAddress(libh, "embed_c"));
        if (!embed_f) {
            std::string error_message = "Failed to load embed function: " + GetLastErrorAsString();
            throw std::runtime_error(error_message);
        }
        get_metadata_f = reinterpret_cast<get_metadata_c_local_func>(GetProcAddress(libh, "get_metadata_c"));
        if (!get_metadata_f) {
            std::string error_message = "Failed to load get_metadata function: " + GetLastErrorAsString();
            throw std::runtime_error(error_message);
        }
        free_metadata_f = reinterpret_cast<free_metadata_c_local_func>(GetProcAddress(libh, "free_metadata_c"));
        if (!free_metadata_f) {
            std::string error_message = "Failed to load free_metadata function: " + GetLastErrorAsString();
            throw std::runtime_error(error_message);
        }
#else
        libh = dlopen(shared_lib_path, RTLD_LAZY);
        if (!libh) {
            std::string error_message = "Failed to load shared library: " + std::string(dlerror());
            throw std::runtime_error(error_message);
        }
        init_embedder_f = reinterpret_cast<init_embedder_local_func>(dlsym(libh, "init_embedder"));
        if (!init_embedder_f) {
            std::string error_message = "Failed to load init_embedder function: " + std::string(dlerror());
            throw std::runtime_error(error_message);
        }
        free_embedder_f = reinterpret_cast<free_embedder_local_func>(dlsym(libh, "free_embedder"));
        if (!free_embedder_f) {
            std::string error_message = "Failed to load free_embedder function: " + std::string(dlerror());
            throw std::runtime_error(error_message);
        }
        embed_f = reinterpret_cast<embed_c_local_func>(dlsym(libh, "embed_c"));
        if (!embed_f) {
            std::string error_message = "Failed to load embed function: " + std::string(dlerror());
            throw std::runtime_error(error_message);
        }
        get_metadata_f = reinterpret_cast<get_metadata_c_local_func>( dlsym(libh, "get_metadata_c"));
        if (!get_metadata_f) {
            std::string error_message = "Failed to load get_metadata function: " + std::string(dlerror());
            throw std::runtime_error(error_message);
        }
        free_metadata_f = reinterpret_cast<free_metadata_c_local_func>(dlsym(libh, "free_metadata_c"));
        if (!free_metadata_f) {
            std::string error_message = "Failed to load free_metadata function: " + std::string(dlerror());
            throw std::runtime_error(error_message);
        }
#endif
        library_ref_count = 1;
        return libh;
    } catch (const std::exception &e) {
        std::string error_message = "Failed to load shared library: " + std::string(e.what());
        set_last_error(error_message.c_str());
        if (libh != nullptr) {

#if defined(_WIN32) || defined(_WIN64)
            if (!FreeLibrary(libh)){
                fprintf(stderr, "Failed to free library %lu\n", GetLastError());
            }
#else
            if(dlclose(libh) != 0){
                fprintf(stderr, "Failed to close library %s\n", dlerror());
            }
#endif
        }
        return nullptr;
    }
}

int init_llama_embedder(char * model_path, uint32_t pooling_type ) {
    std::lock_guard<std::mutex> lock(embedder_mutex);
    if (!libh) {
        set_last_error("Shared library not loaded, use load_library first.");
        return -1;
    }
    try {
        embedder = init_embedder_f(model_path, pooling_type);
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
    std::lock_guard<std::mutex> lock(embedder_mutex);
    if (library_ref_count.fetch_sub(1) == 1) {
        if (embedder != nullptr) {
            free_embedder_f(embedder);
        }
        if (libh != nullptr) {
#if defined(_WIN32) || defined(_WIN64)
            std::this_thread::sleep_for(std::chrono::milliseconds(100)); //small delay to allow for backend to be freed
            if (!FreeLibrary(libh)){
                fprintf(stderr, "Failed to free library %lu\n", GetLastError());
            }
#else
            if (dlclose(libh) != 0) {
                fprintf(stderr, "Failed to close library %s\n", dlerror());
            }
#endif
        }
    }
}

FloatMatrix llama_embedder_embed(const char** texts, size_t text_count, int32_t norm) {
    std::lock_guard<std::mutex> lock(embedder_mutex);
    FloatMatrix fm = {nullptr, 0, 0};
    try {
        std::vector<std::vector<float>> output;
        fm = embed_f(embedder, texts, text_count, norm);
    } catch (const std::exception &e) {
        set_last_error(e.what());
    }
    return fm;
}

char** llama_embedder_get_metadata(size_t* size) {
    std::lock_guard<std::mutex> lock(embedder_mutex);
    MetadataPair* metadata_array = nullptr;
    char** metadata = nullptr;
    *size = 0;

    if (get_metadata_f(embedder, &metadata_array, size) != 0 || metadata_array == nullptr) {
        fprintf(stderr, "Failed to get metadata\n");
        return nullptr;
    }

    metadata = (char**)malloc(*size * sizeof(char*));
    if (metadata == nullptr) {
        fprintf(stderr, "Failed to allocate memory for metadata\n");
        free(metadata_array);
        *size = 0;
        return nullptr;
    }

    for (size_t i = 0; i < *size; i++) {
        if (metadata_array[i].key == nullptr || metadata_array[i].value == nullptr) {
            fprintf(stderr, "Null key or value at index %zu\n", i);
            continue;
        }

        size_t key_len = strlen(metadata_array[i].key);
        size_t value_len = strlen(metadata_array[i].value);
        metadata[i] = (char*)malloc(key_len + value_len + 2); // +2 for '=' and null terminator
        if (metadata[i] == nullptr) {
            fprintf(stderr, "Failed to allocate memory for metadata[%zu]\n", i);
            continue;
        }

        snprintf(metadata[i], key_len + value_len + 2, "%s=%s", metadata_array[i].key, metadata_array[i].value);

    }
    free_metadata_f(metadata_array, *size);

    return metadata;
}

void free_metadata(char** metadata_array, size_t size) {
    for (size_t i = 0; i < size; i++) {
        free(metadata_array[i]);
    }
    free(metadata_array);
}

void free_float_matrixw(FloatMatrix * fm) {
    if (fm != nullptr){
        if (fm->data != nullptr) {
            free(fm->data);
            fm->data = nullptr;
        }
    }
}


}
