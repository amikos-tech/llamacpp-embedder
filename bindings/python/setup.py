import os
import platform
import shutil
from typing import List

import pybind11
from setuptools import Extension, setup, find_packages
from setuptools.command.build_ext import build_ext
from setuptools.command.sdist import sdist

__version__ = "0.0.1"


def get_lib_name():
    if platform.system() == "Darwin":
        return "libllama-embedder.dylib"
    elif platform.system() == "Linux":
        return "libllama-embedder.so"
    elif platform.system() == "Windows":
        return ["llama-embedder.dll","llama-embedder.lib"]
    else:
        raise OSError(f"Unsupported operating system: {platform.system()}")


shared_lib_target = "../../build"
if platform.system() == "Windows":
    shared_lib_target = "../../build/Release"
# Define the path to your shared library relative to the project root
if isinstance(get_lib_name(),str):
    SHARED_LIB_PATHS = os.path.join(os.environ.get("SHARED_LIB_PATH", shared_lib_target), get_lib_name())
else:
    SHARED_LIB_PATHS = [os.path.join(os.environ.get("SHARED_LIB_PATH", shared_lib_target), lib) for lib in get_lib_name()]
SHARED_LIB_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src"))
LICENSE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../LICENSE.md"))
LLAMA_LICENSE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../vendor/llama.cpp/LICENSE"))


class CustomBuildExt(build_ext):
    def run(self):
        # Ensure the shared library exists in the current directory
        if not os.path.exists(get_lib_name()):
            raise FileNotFoundError(f"Shared library not found at {get_lib_name()}")

        # Copy the shared library to the build directory
        dest_path = os.path.join(self.build_lib, "llama_embedder")

        os.makedirs(dest_path, exist_ok=True)
        shutil.copy2(get_lib_name(), dest_path)

        # Run the original build_ext command
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


class CustomSdist(sdist):
    """
    Here we create the release tree by adding the necessary build deps such as the shared lib and the src or header files
    """

    def make_release_tree(self, base_dir, files):
        sdist.make_release_tree(self, base_dir, files)
        # Copy shared library to the base dir of the source distribution
        if isinstance(SHARED_LIB_PATHS, List):
            for SHARED_LIB_PATH in SHARED_LIB_PATHS:
                if os.path.exists(SHARED_LIB_PATH):
                    dest = os.path.join(base_dir, os.path.basename(SHARED_LIB_PATH))
                    shutil.copy2(SHARED_LIB_PATH, dest)
                else:
                    raise FileNotFoundError(
                        f"Shared library not found at {SHARED_LIB_PATH}, {os.listdir(os.path.dirname(SHARED_LIB_PATH))}")
        else:
            dest = os.path.join(base_dir, os.path.basename(SHARED_LIB_PATHS))
            shutil.copy2(SHARED_LIB_PATHS, dest)
        if os.path.exists(SHARED_LIB_SRC):
            dest_src_path = os.path.join(base_dir, "src")
            shutil.copytree(SHARED_LIB_SRC, dest_src_path, dirs_exist_ok=True)
        if os.path.exists(LICENSE_PATH):
            shutil.copy2(LICENSE_PATH, base_dir)
        if os.path.exists(LLAMA_LICENSE_PATH):
            os.makedirs(os.path.join(base_dir, "vendor/llama.cpp"), exist_ok=True)
            shutil.copy2(LLAMA_LICENSE_PATH, os.path.join(base_dir, "vendor/llama.cpp/LICENSE"))


ext_modules = [
    Extension(
        "llama_embedder",
        ["bindings.cpp"],
        include_dirs=[
            pybind11.get_include(),
            ".",
        ],
        libraries=["llama-embedder"],  # Link against the shared library without the 'lib' prefix and extension
        language="c++",
        extra_link_args=[
            "-L" + os.getcwd(),  # Explicitly specify the directory containing the dylib
            "-lllama-embedder",  # Ensure the correct library is linked
            "-Wl,-rpath,@loader_path/"  # Set rpath to the directory containing the .so and .dylib
        ],
    ),
]
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="llama_embedder",
    version=__version__,
    description="LLama.cpp embedder library python bindings",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Trayan Azarov (Amikos Tech)",
    author_email="trayan.azarov@amikos.tech",
    maintainer_email="trayan.azarov@amikos.tech",
    url="https://github.com/amikos-tech/llamacpp-embedder",
    packages=find_packages(),
    ext_modules=ext_modules,
    install_requires=["pybind11>=2.6.0"],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: System :: Archiving :: Packaging",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    cmdclass={
        "build_ext": CustomBuildExt,
        "sdist": CustomSdist,
    },
    package_data={"llama_embedder": [get_lib_name()]},
    include_package_data=True,
    zip_safe=False,
)
