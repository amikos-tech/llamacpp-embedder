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
      "cflags!": [ "" ],
      "cflags_cc!": [ "" , "-std=c++11" , "-Wall", "-Wextra", "-pedantic"],
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