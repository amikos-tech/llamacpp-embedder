#include <napi.h>
#include "embedder.h" // Make sure this header is in the include path

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

    std::vector<std::string> texts;
    for (uint32_t i = 0; i < texts_array.Length(); i++) {
        Napi::Value text = texts_array[i];
        if (!text.IsString()) {
            Napi::TypeError::New(env, "Array must contain only strings").ThrowAsJavaScriptException();
            return env.Null();
        }
        texts.push_back(text.As<Napi::String>().Utf8Value());
    }

    std::vector<std::vector<float>> embeddings;
    try {
        embed(this->embedder, texts, embeddings, embd_norm);
    } catch (const std::exception& e) {
        Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
        return env.Null();
    }

    Napi::Array result = Napi::Array::New(env, embeddings.size());
    for (size_t i = 0; i < embeddings.size(); i++) {
        Napi::Array embedding = Napi::Array::New(env, embeddings[i].size());
        for (size_t j = 0; j < embeddings[i].size(); j++) {
            embedding[j] = Napi::Number::New(env, embeddings[i][j]);
        }
        result[i] = embedding;
    }

    return result;
}

Napi::Value LlamaEmbedder::GetMetadata(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    std::unordered_map<std::string, std::string> metadata;
    try {
        get_metadata(this->embedder, metadata);
    } catch (const std::exception& e) {
        Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
        return env.Null();
    }

    Napi::Object result = Napi::Object::New(env);
    for (const auto& pair : metadata) {
        result.Set(pair.first, pair.second);
    }

    return result;
}

Napi::Object Init(Napi::Env env, Napi::Object exports) {
    return LlamaEmbedder::Init(env, exports);
}

NODE_API_MODULE(llama_embedder, Init)
