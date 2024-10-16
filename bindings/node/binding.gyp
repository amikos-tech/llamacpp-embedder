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
      "defines": [ "NAPI_CPP_EXCEPTIONS" ],
      "cflags!": [ "-fno-exceptions" ],
      "cflags_cc!": [ "" , "-std=c++11" , "-Wall", "-Wextra", "-pedantic","-fno-exceptions"],
      "xcode_settings": {
        "GCC_ENABLE_CPP_EXCEPTIONS": "YES",
        "CLANG_CXX_LIBRARY": "libc++",
        "MACOSX_DEPLOYMENT_TARGET": "10.9"
      },
      "msvs_settings": {
        "VCCLCompilerTool": { "ExceptionHandling": 1 }
      },
      "conditions": [
      [ 'OS=="mac"', {
        "libraries": [
          "-framework Accelerate",
          "-framework Metal",
          "-framework Foundation",
          "-framework MetalKit"
        ]
      }],
      [ 'OS=="linux"', {
        "libraries": [
          "-fopenmp",
        ]
      }],
      ],
      "libraries": [
        "../../../build/static/libllama-embedder.a",
        "../../../build/static/libllama.a",
        "../../../build/static/libggml.a",
        "../../../build/static/libcommon.a",
      ]
    }
  ]
}