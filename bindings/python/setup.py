import os
import platform
import shutil

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
import pybind11
from setuptools.command.sdist import sdist
from wheel._bdist_wheel import bdist_wheel

SHARED_LIB_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src"))
LICENSE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../LICENSE.md"))
LLAMA_LICENSE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../vendor/llama.cpp/LICENSE"))


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

SHARED_LIB_PATH = os.path.join('../../build', get_lib_name())

class CustomBuildExt(build_ext):
    def run(self):
        print("doing CustomBuildExt")
        # Use environment variable set by cibuildwheel, or fall back to a default
        shared_lib_path = os.path.join(get_lib_name())
        print(f"{os.listdir(self.build_lib)}", f"{os.path.abspath(self.build_lib)}")
        print(f"Looking for shared library at: {shared_lib_path}")

        if not os.path.exists(shared_lib_path):
            raise FileNotFoundError(f"Shared library not found at {shared_lib_path}")

        dest_path = os.path.join(self.build_lib)
        os.makedirs(dest_path, exist_ok=True)
        self.copy_file(shared_lib_path, os.path.join(self.build_lib, os.path.basename(shared_lib_path)))

        build_ext.run(self)

    def build_extensions(self):
        print("doing build_extensions")
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
        # library_dirs=["."],  # Adjust this path to point to your built libraries
        libraries=["llama-embedder"],
        language="c++",
        extra_link_args=[
            "-L" + os.getcwd(),  # Explicitly specify the directory containing the dylib
            "-lllama-embedder",  # Ensure the correct library is linked
            "-Wl,-rpath,@loader_path/"  # Set rpath to the directory containing the .so and .dylib
        ],
    ),
]


class CustomSdist(sdist):
    """
    Here we create the release tree by adding the necessary build deps such as the shared lib and the src or header files
    """

    def make_release_tree(self, base_dir, files):
        print("doing make_release_tree")
        sdist.make_release_tree(self, base_dir, files)
        # Copy shared library to the base dir of the source distribution
        dest = os.path.join(base_dir, os.path.basename(SHARED_LIB_PATH))
        shutil.copy2(SHARED_LIB_PATH, dest)
        if os.path.exists(SHARED_LIB_SRC):
            dest_src_path = os.path.join(base_dir, "src")
            shutil.copytree(SHARED_LIB_SRC, dest_src_path, dirs_exist_ok=True)
        if os.path.exists(LICENSE_PATH):
            shutil.copy2(LICENSE_PATH, base_dir)
        if os.path.exists(LLAMA_LICENSE_PATH):
            os.makedirs(os.path.join(base_dir, "vendor/llama.cpp"), exist_ok=True)
            shutil.copy2(LLAMA_LICENSE_PATH, os.path.join(base_dir, "vendor/llama.cpp/LICENSE"))


class CustomBdistWheel(bdist_wheel):
    """
    Here we create the release tree by adding the necessary build deps such as the shared lib and the src or header files
    """

    def run(self):
        print("doing CustomBdistWheel")

        # Custom behavior before the standard run
        print("Running custom bdist_wheel command")



        # Copy shared library to the base dir of the source distribution
        base = "."
        print(f"Project root dir: {os.path.abspath(base)}: {os.listdir(base)}")
        _shared_lib=os.path.join(base, get_lib_name())
        _src_path = os.path.join(base, "src")
        _license_path = os.path.join(base, "LICENSE.md")
        _llama_license_path = os.path.join(base, "vendor/llama.cpp/LICENSE")
        dest = os.path.join(self.get_wheel_dir(), os.path.basename(_shared_lib))
        shutil.copy2(_shared_lib, dest)
        if os.path.exists(_src_path):
            dest_src_path = os.path.join(self.dist_dir, "src")
            shutil.copytree(_src_path, dest_src_path, dirs_exist_ok=True)
        if os.path.exists(_license_path):
            shutil.copy2(_license_path, self.dist_dir)
        if os.path.exists(_llama_license_path):
            os.makedirs(os.path.join(self.dist_dir, "vendor/llama.cpp"), exist_ok=True)
            shutil.copy2(_llama_license_path, os.path.join(self.dist_dir, "vendor/llama.cpp/LICENSE"))
        # Call the standard run method
        super().run()


setup(
    package_data={"llama_embedder": [get_lib_name()]},
    include_package_data=True,
    zip_safe=False,
    ext_modules=ext_modules,
    cmdclass={"build_ext": CustomBuildExt,"sdist": CustomSdist,"bdist_wheel": CustomBdistWheel,},
)
