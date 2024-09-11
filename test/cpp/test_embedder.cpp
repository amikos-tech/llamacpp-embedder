//
// Created by Trayan Azarov on 11.09.24.
//

#include "test_embedder.h"

#include "gtest/gtest.h"
#include "../../src/embedder.h" // Include the header file where the code is defined
// Function to test
int add(int a, int b) {
    return a + b;
}

TEST(EmbedderTest, InitWithValidModel) {
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

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}