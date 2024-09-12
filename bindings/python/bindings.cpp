#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../../src/embedder.h"

namespace py = pybind11;

enum class NormalizationType {
    NONE = -1,
    MAX_ABS_INT16 = 0,
    TAXICAB = 1,
    EUCLIDEAN = 2,
    // >2 = p-norm
};

enum class PoolingType {
    NONE = 0,
    MEAN = 1,
    CLS = 2,
    LAST = 3,
};

class TokenizerData {
public:
    std::vector<int32_t> tokens;
    std::vector<int32_t> attention_mask;

    TokenizerData(const std::vector<int32_t>& tokens, const std::vector<int32_t>& attention_mask) : tokens(tokens), attention_mask(attention_mask) {}
};


class LlamaEmbedder {
private:
    llama_embedder* embedder;

public:
    LlamaEmbedder(const std::string& model_path, const PoolingType pooling_type = PoolingType::MEAN) {

        embedder = init_embedder(const_cast<char*>(model_path.c_str()), static_cast<uint32_t>(pooling_type));
        if (!embedder) {
            throw std::runtime_error("Failed to initialize embedder");
        }
    }

    ~LlamaEmbedder() {
        if (embedder) {
            free_embedder(embedder);
        }
    }

    std::vector<std::vector<float>> embed(const std::vector<std::string>& texts, NormalizationType norm) {
        if (!embedder) {
            throw std::runtime_error("Embedder is not initialized");
        }

        if (texts.empty()) {
            throw std::runtime_error("Texts are empty");
        }
        std::vector<std::vector<float>> output;
        ::embed(embedder, texts, output, static_cast<int32_t>(norm));
        return output;
    }

    std::unordered_map<std::string, std::string> get_metadata() {
        if (!embedder) {
            throw std::runtime_error("Embedder is not initialized");
        }
        std::unordered_map<std::string, std::string> metadata;
        ::get_metadata(embedder, metadata);
        return metadata;
    }

    std::vector<TokenizerData> tokenize(std::vector<std::string>& texts, const bool add_special_tokens = true, const bool parse_special = false, const bool enable_padding = false) {
        std::vector<TokenizerData> final_output;
        std::vector<llama_tokenizer_data> output;
        if (!embedder) {
            throw std::runtime_error("Embedder is not initialized");
        }
        if (texts.empty()) {
            throw std::runtime_error("Texts are empty");
        }
        ::tokenize(embedder, texts, output, add_special_tokens, parse_special, enable_padding);

        for (const auto& tokenizer_data : output) {
            TokenizerData temp(tokenizer_data.tokens, tokenizer_data.attention_mask);
            final_output.push_back(temp);
        }
        return final_output;
    }
};

PYBIND11_MODULE(llama_embedder, m) {
m.doc() = "Python bindings for llama-embedder";

py::enum_<NormalizationType>(m, "NormalizationType")
.value("NONE", NormalizationType::NONE)
.value("MAX_ABS_INT16", NormalizationType::MAX_ABS_INT16)
.value("TAXICAB", NormalizationType::TAXICAB)
.value("EUCLIDEAN", NormalizationType::EUCLIDEAN)
.export_values();

py::enum_<PoolingType>(m, "PoolingType")
.value("NONE", PoolingType::NONE)
.value("MEAN", PoolingType::MEAN)
.value("CLS", PoolingType::CLS)
.value("LAST", PoolingType::LAST)
.export_values();

py::class_<TokenizerData>(m, "TokenizerData")
.def(py::init<const std::vector<int32_t>&, const std::vector<int32_t>&>(), py::arg("tokens"), py::arg("attention_mask"))
.def_readwrite("tokens", &TokenizerData::tokens)  // Bind tokens attribute
.def_readwrite("attention_mask", &TokenizerData::attention_mask);  // Bind attention_mask attribute


py::class_<LlamaEmbedder>(m, "LlamaEmbedder")
.def(py::init<const std::string&, PoolingType>(), py::arg("model_path"), py::arg("pooling_type") = PoolingType::MEAN)  // Updated init
.def("embed", &LlamaEmbedder::embed, "Create embeddings from prompts",
py::arg("texts"), py::arg("norm") = NormalizationType::EUCLIDEAN)
.def("get_metadata", &LlamaEmbedder::get_metadata, "Get metadata of the model")
.def("tokenize", &LlamaEmbedder::tokenize, "Tokenize the input texts",py::arg("texts"), py::arg("add_special_tokens") = true, py::arg("parse_special") = false, py::arg("enable_padding") = false)
.def("__enter__", [](LlamaEmbedder& self) { return &self; })
.def("__exit__", [](LlamaEmbedder& self, py::object exc_type, py::object exc_value, py::object traceback) {});
}