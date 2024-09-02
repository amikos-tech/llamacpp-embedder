import os
import platform
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
import pybind11

def get_lib_name():
    if platform.system() == "Darwin":
        return "libllama-embedder.dylib"
    elif platform.system() == "Linux":
        return "libllama-embedder.so"
    elif platform.system() == "Windows":
        # return ["llama-embedder.dll","llama-embedder.lib"]
        return "llama-embedder.dll"
    else:
        raise OSError(f"Unsupported operating system: {platform.system()}")

class CustomBuildExt(build_ext):
    def run(self):
        # Use environment variable set by cibuildwheel, or fall back to a default
        shared_lib_path = os.path.join('build', get_lib_name())
        print(f"Looking for shared library at: {shared_lib_path}")

        if not os.path.exists(shared_lib_path):
            raise FileNotFoundError(f"Shared library not found at {shared_lib_path}")

        dest_path = os.path.join(self.build_lib, "llama_embedder")
        os.makedirs(dest_path, exist_ok=True)
        self.copy_file(shared_lib_path, os.path.join(dest_path, os.path.basename(shared_lib_path)))

        build_ext.run(self)

    def build_extensions(self):
        # Determine the compiler and set appropriate flags
        ct = self.compiler.compiler_type
        opts = []
        if ct == "unix":
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
            opts.append("-std=c++11")
            if platform.system() == "Darwin":
                opts.extend(["-stdlib=libc++", "-mmacosx-version-min=10.7"])
        elif ct == "msvc":
            opts.append(f'/DVERSION_INFO=\\"{self.distribution.get_version()}\\"')

        for ext in self.extensions:
            ext.extra_compile_args = opts
        build_ext.build_extensions(self)

ext_modules = [
    Extension(
        "llama_embedder",
        ["bindings.cpp"],
        include_dirs=[
            pybind11.get_include(),
            ".",
            "src",  # Adjust this path to point to your C++ headers
        ],
        library_dirs=["build"],  # Adjust this path to point to your built libraries
        libraries=["llama-embedder"],
        language="c++",
        extra_link_args=["-Wl,-rpath,@loader_path/"],
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": CustomBuildExt},
)
