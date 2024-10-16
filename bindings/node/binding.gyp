{
  "targets": [
    {
      "target_name": "llama_embedder",
      "sources": [ "llama_embedder.cpp" ],
      "include_dirs": [
        "<!@(node -p \"require('node-addon-api').include\")",
        "../../src"
      ],
      "dependencies": [
        "<!(node -p \"require('node-addon-api').gyp\")"
      ],
      "defines": [
          "NAPI_CPP_EXCEPTIONS",
          "LLAMA_EMBEDDER_STATIC",
      ],
      "cflags!": [ "-fno-exceptions" ],
      "cflags_cc!": [ "" , "-std=c++11" , "-Wall", "-Wextra", "-pedantic","-fno-exceptions"],
      "xcode_settings": {
        "GCC_ENABLE_CPP_EXCEPTIONS": "YES",
        "CLANG_CXX_LIBRARY": "libc++",
        "MACOSX_DEPLOYMENT_TARGET": "10.9"
      },
      "msvs_settings": {
        "VCCLCompilerTool": {
          "ExceptionHandling": 1,
          "AdditionalOptions": [ "/MD" ]
        },
      },
      "conditions": [
      [ 'OS=="mac"', {
        "libraries": [
          "-framework Accelerate",
          "-framework Metal",
          "-framework Foundation",
          "-framework MetalKit",
          "../../../build/static/libllama-embedder.a",
          "../../../build/static/libllama.a",
          "../../../build/static/libggml.a",
          "../../../build/static/libcommon.a",
        ]
      }],
      [ 'OS=="linux"', {
        "libraries": [
          "-fopenmp",
          "../../../build/static/libllama-embedder.a",
          "../../../build/static/libllama.a",
          "../../../build/static/libggml.a",
          "../../../build/static/libcommon.a",
        ]
      }],
        [ 'OS=="win"', {
            "libraries": [
            "../../../build/static/llama-embedder.lib",
            "../../../build/static/llama.lib",
            "../../../build/static/ggml.lib",
            "../../../build/static/common.lib",
            "kernel32.lib"
            ]
        }]
      ],
    }
  ]
}