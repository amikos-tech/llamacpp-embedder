#include <napi.h>
#include "embedder.h" // Make sure this header is in the include path
#include <stdexcept>

class LlamaEmbedder : public Napi::ObjectWrap<LlamaEmbedder> {
public:
    static Napi::Object Init(Napi::Env env, Napi::Object exports);
    LlamaEmbedder(const Napi::CallbackInfo& info);
    ~LlamaEmbedder();

private:
    static Napi::FunctionReference constructor;
    llama_embedder* embedder;

    Napi::Value Embed(const Napi::CallbackInfo& info);
    Napi::Value GetMetadata(const Napi::CallbackInfo& info);
};

Napi::FunctionReference LlamaEmbedder::constructor;

Napi::Object LlamaEmbedder::Init(Napi::Env env, Napi::Object exports) {
    Napi::Function func = DefineClass(env, "LlamaEmbedder", {
            InstanceMethod("embed", &LlamaEmbedder::Embed),
            InstanceMethod("getMetadata", &LlamaEmbedder::GetMetadata)
    });

    constructor = Napi::Persistent(func);
    constructor.SuppressDestruct();

    exports.Set("LlamaEmbedder", func);
    return exports;
}

LlamaEmbedder::LlamaEmbedder(const Napi::CallbackInfo& info) : Napi::ObjectWrap<LlamaEmbedder>(info) {
    Napi::Env env = info.Env();

    if (info.Length() < 2) {
        Napi::TypeError::New(env, "Wrong number of arguments").ThrowAsJavaScriptException();
        return;
    }

    if (!info[0].IsString() || !info[1].IsNumber()) {
        Napi::TypeError::New(env, "Wrong arguments").ThrowAsJavaScriptException();
        return;
    }

    std::string model_path = info[0].As<Napi::String>().Utf8Value();
    uint32_t pooling_type = info[1].As<Napi::Number>().Uint32Value();

    try {
        this->embedder = init_embedder(model_path.c_str(), pooling_type);
    } catch (const std::exception& e) {
        Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
    }
}

LlamaEmbedder::~LlamaEmbedder() {
    if (this->embedder) {
        free_embedder(this->embedder);
    }
}

Napi::Value LlamaEmbedder::Embed(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 2) {
        Napi::TypeError::New(env, "Wrong number of arguments").ThrowAsJavaScriptException();
        return env.Null();
    }

    if (!info[0].IsArray() || !info[1].IsNumber()) {
        Napi::TypeError::New(env, "Wrong arguments").ThrowAsJavaScriptException();
        return env.Null();
    }

    Napi::Array texts_array = info[0].As<Napi::Array>();
    int32_t embd_norm = info[1].As<Napi::Number>().Int32Value();

    std::vector<std::string> texts(texts_array.Length());
    std::vector<const char*> texts_c(texts_array.Length());
    for (uint32_t i = 0; i < texts_array.Length(); i++) {
        Napi::Value text = texts_array[i];
        if (!text.IsString()) {
            Napi::TypeError::New(env, "Array must contain only strings").ThrowAsJavaScriptException();
            return env.Null();
        }
        texts[i] = text.As<Napi::String>().Utf8Value();
        texts_c[i] = texts[i].c_str();
    }

    FloatMatrix embeddings;
    try {
        embeddings = embed_c(this->embedder, texts_c.data(), texts.size(), embd_norm);
    } catch (const std::exception& e) {
        Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
        return env.Null();
    }

    Napi::Array result = Napi::Array::New(env, embeddings.rows);
    for (size_t i = 0; i < embeddings.rows; i++) {
        Napi::Array embedding = Napi::Array::New(env, embeddings.cols);
        for (size_t j = 0; j < embeddings.cols; j++) {
            embedding[j] = Napi::Number::New(env, *(embeddings.data + i * embeddings.cols + j));
        }
        result[i] = embedding;
    }
    free_float_matrix(&embeddings);
    return result;
}

Napi::Value LlamaEmbedder::GetMetadata(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    MetadataPair* metadata_array = nullptr;
    size_t count = 0;

    try {
        int result = get_metadata_c(this->embedder, &metadata_array, &count);
        if (result != 0) {
            throw std::runtime_error("Failed to get metadata");
        }

        Napi::Object result_obj = Napi::Object::New(env);
        for (size_t i = 0; i < count; i++) {
            result_obj.Set(metadata_array[i].key, metadata_array[i].value);
        }

        // Free the allocated memory
        free_metadata_c(metadata_array, count);

        return result_obj;
    } catch (const std::exception& e) {
        // Make sure to free the memory even if an exception occurs
        if (metadata_array != nullptr) {
            free_metadata_c(metadata_array, count);
        }
        Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
        return env.Null();
    }
}

Napi::Object Init(Napi::Env env, Napi::Object exports) {
    return LlamaEmbedder::Init(env, exports);
}

NODE_API_MODULE(llama_embedder, Init)
