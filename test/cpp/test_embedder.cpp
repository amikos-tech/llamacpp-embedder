//
// Created by Trayan Azarov on 11.09.24.
//

#include "test_embedder.h"
#include <iostream>
#include "gtest/gtest.h"
#include "../../src/embedder.h" // Include the header file where the code is defined

void printVector(const std::vector<int>& vec) {
    for (int num : vec) {
        std::cout << num << " ";
    }
    std::cout << std::endl;
}

uint32_t get_mask_size(const std::vector<int>& tokens) {
    int mask_size = 0;
    for (int i = tokens.size() - 1; i >= 0; i--) {
        if (tokens[i] != 0) {
            mask_size++;
        }
    }
    return mask_size;
}

TEST(EmbedderTest, InitWithModel) {
const char* valid_model_path = "snowflake-arctic-embed-s/snowflake-arctic-embed-s-f16.GGUF";
uint32_t pooling_type = 1; // LLAMA_POOLING_TYPE_NONE
llama_embedder* embedder = init_embedder(valid_model_path, pooling_type);
EXPECT_NE(embedder, nullptr);
EXPECT_NE(embedder->context, nullptr);
EXPECT_NE(embedder->model, nullptr);

free_embedder(embedder);
}

TEST(EmbedderTest, EmbedWithModel) {
const char* valid_model_path = "snowflake-arctic-embed-s/snowflake-arctic-embed-s-f16.GGUF";
uint32_t pooling_type = 1; // LLAMA_POOLING_TYPE_NONE
llama_embedder* embedder = init_embedder(valid_model_path, pooling_type);

std::vector<std::vector<float>> output;
embed(embedder, std::vector<std::string>{"Hello, world!"}, output, 2);
EXPECT_NE(embedder, nullptr);
EXPECT_NE(embedder->context, nullptr);
EXPECT_NE(embedder->model, nullptr);
EXPECT_EQ(output.size(), 1);
EXPECT_EQ(output[0].size(), 384);

free_embedder(embedder);
}

TEST(EmbedderTest, EmbedWithModelNoTextsError) {
const char* valid_model_path = "snowflake-arctic-embed-s/snowflake-arctic-embed-s-f16.GGUF";
uint32_t pooling_type = 1; // LLAMA_POOLING_TYPE_NONE
llama_embedder* embedder = init_embedder(valid_model_path, pooling_type);

std::vector<std::vector<float>> output;
embed(embedder, std::vector<std::string>{}, output, 2);
EXPECT_NE(embedder, nullptr);
EXPECT_NE(embedder->context, nullptr);
EXPECT_NE(embedder->model, nullptr);
EXPECT_EQ(output.size(), 0);

free_embedder(embedder);
}

TEST(EmbedderTest, GetMetadata) {
const char* valid_model_path = "snowflake-arctic-embed-s/snowflake-arctic-embed-s-f16.GGUF";
uint32_t pooling_type = 1; // LLAMA_POOLING_TYPE_NONE
llama_embedder* embedder = init_embedder(valid_model_path, pooling_type);

std::unordered_map<std::string, std::string> metadata = {};
get_metadata(embedder, metadata);

EXPECT_NE(embedder, nullptr);
EXPECT_NE(embedder->context, nullptr);
EXPECT_NE(embedder->model, nullptr);
EXPECT_FALSE(metadata.empty()); // Check if the map is empty
EXPECT_EQ(metadata["general.name"], "snowflake-arctic-embed-s");
EXPECT_EQ(metadata["general.architecture"], "bert");

free_embedder(embedder);
}

TEST(EmbedderTest, Tokenize) {
const char* valid_model_path = "snowflake-arctic-embed-s/snowflake-arctic-embed-s-f16.GGUF";
uint32_t pooling_type = 1; // LLAMA_POOLING_TYPE_NONE
llama_embedder* embedder = init_embedder(valid_model_path, pooling_type);
std::vector<std::string> prompts = {"Hello, world!", "How are you?"};
std::vector<llama_tokenizer_data> output;

tokenize(embedder, prompts, output);

EXPECT_NE(embedder, nullptr);
EXPECT_NE(embedder->context, nullptr);
EXPECT_NE(embedder->model, nullptr);
EXPECT_FALSE(output.empty());
EXPECT_EQ(output.size(), 2);

for (const auto& tokenizer_data : output) {
    EXPECT_EQ(tokenizer_data.tokens.size(), 512);
    EXPECT_EQ(tokenizer_data.attention_mask.size(), 512);
    EXPECT_EQ(get_mask_size(tokenizer_data.tokens), 6); // this works for the given prompts
}

free_embedder(embedder);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}