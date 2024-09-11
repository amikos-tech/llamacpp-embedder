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

    std::vector<std::vector<float>> embed(const std::vector<std::string>& prompts, NormalizationType norm) {
        std::vector<std::vector<float>> output;
        if (!embedder) {
            throw std::runtime_error("Embedder is not initialized");
        }
        ::embed(embedder, prompts, output, static_cast<int32_t>(norm));
        return output;
    }

    std::unordered_map<std::string, std::string> get_metadata() {
        std::unordered_map<std::string, std::string> metadata;
        if (!embedder) {
            throw std::runtime_error("Embedder is not initialized");
        }
        ::get_metadata(embedder, metadata);
        return metadata;
    }

    std::vector<std::unordered_map<std::string, std::vector<int32_t>>> tokenize(std::vector<std::string>& texts) {
        std::vector<std::unordered_map<std::string, std::vector<int32_t>>> final_output;
        std::vector<llama_tokenizer_data> output;
        if (!embedder) {
            throw std::runtime_error("Embedder is not initialized");
        }
        ::tokenize(embedder, texts, output);

        for (const auto& tokenizer_data : output) {
            std::unordered_map<std::string, std::vector<int32_t>> temp;
            temp["tokens"] = tokenizer_data.tokens;
            temp["attention_mask"] = tokenizer_data.attention_mask;
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

py::class_<LlamaEmbedder>(m, "LlamaEmbedder")
.def(py::init<const std::string&, PoolingType>(), py::arg("model_path"), py::arg("pooling_type") = PoolingType::MEAN)  // Updated init
.def("embed", &LlamaEmbedder::embed, "Create embeddings from prompts",
py::arg("prompts"), py::arg("norm") = NormalizationType::EUCLIDEAN)
.def("get_metadata", &LlamaEmbedder::get_metadata, "Get metadata of the model")
.def("tokenize", &LlamaEmbedder::tokenize, "Tokenize the input texts")
.def("__enter__", [](LlamaEmbedder& self) { return &self; })
.def("__exit__", [](LlamaEmbedder& self, py::object exc_type, py::object exc_value, py::object traceback) {});
}