# Llama Embedder Server

A simple server to serve embedding inference using llama.cpp models (GGUF).

## Up and Running

```bash
go install github.com/amikos-tech/llamacpp-embedder/server
export LLAMA_CACHE_DIR=./cache # set model cache directory
export LLAMA_MODEL_TTL_MINUTES=60 # 1 hour
export LLAMA_CACHED_MODELS="ChristianAzinn/snowflake-arctic-embed-s-gguf/snowflake-arctic-embed-s-f16.GGUF;leliuga/all-MiniLM-L6-v2-GGUF/all-MiniLM-L6-v2.Q4_0.gguf" # list of models to cache (requires internet connection)
llama-embedder-server
```

### Endpoints

- `/embed_texts` - POST - Embed a list of texts
- `/embed_models` - GET - List of cached models
- `/version` - GET - Server version
- `/health` - GET - Server health

### Environment Variables

- `LLAMA_CACHE_DIR` - Directory to cache models (default: `~/./cache/llama_cache`)
- `LLAMA_MODEL_TTL_MINUTES` - TTL for cached models in minutes (default: `60`)
- `LLAMA_CACHED_MODELS` - List of models to cache. If the models are not cached, the server will download them from Hugging Face Hub.

## Debug info

Profiling the binding:

Create a main.cpp in `internal/embedder/`:

```cpp
#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <chrono>
#include "wrapper.h"

int main() {
    const char* model_path = "/path/to/model/all-MiniLM-L6-v2.Q4_0.gguf";  // Replace with actual path
    uint32_t pooling_type = 1;  // Replace with actual pooling type if needed

    llama_embedder* embedder = nullptr;

    // Initialize embedder
    if (init_embedder_l(&embedder, model_path, pooling_type) != 0) {
        std::cerr << "Failed to initialize embedder: " << get_last_error() << std::endl;
        return 1;
    }

    std::cout << "Embedder initialized successfully." << std::endl;

    // Prepare test texts
        const char* texts_array[] = {
                "Hello, world!",
                "This is a test.",
                "Embedding some text."
            };
        size_t text_count = sizeof(texts_array) / sizeof(texts_array[0]);
        const char** texts = texts_array;
        while (true) {
                std::cout << "Embedding." << std::endl;

                // Embed texts
                FloatMatrixW result = embed_texts(embedder, texts, text_count, 2);
                if (result.data == nullptr) {
                    std::cerr << "Failed to embed texts: " << get_last_error() << std::endl;
                    free_embedder_l(embedder);
                    return 1;
                }

                std::cout << "Texts embedded successfully." << std::endl;
                std::cout << "Result dimensions: " << result.rows << " x " << result.cols << std::endl;

                // Print first few values of the first embedding
                std::cout << "First few values of the first embedding:" << std::endl;
                for (size_t i = 0; i < std::min(result.cols, size_t(5)); ++i) {
                    std::cout << result.data[i] << " ";
                }
                std::cout << std::endl;

                // Clean up
                free_float_matrixw(&result);

                // Sleep for 1 second
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }

            // Clean up embedder
            free_embedder_l(embedder);

            return 0;
}
```

Compile the static libraries and then compile the main.cpp:

```bash
g++ -o embedder_test main.cpp wrapper.cpp  -I. -I../../../src -std=c++14 -Wall -Wextra -pedantic -L../../../build/static -lllama-embedder -lcommon -lllama -lggml -framework Accelerate -framework Metal -framework Foundation -framework MetalKit
```

Adjust the above command to match your OS (the above is for macOS).

Run the compiled binary:

```bash
./embedder_test
```

Check for leaks:
```bash
leaks <process_id> > leaks.txt
```