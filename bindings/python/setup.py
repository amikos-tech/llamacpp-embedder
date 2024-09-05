import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List

from setuptools import Extension, setup, glob
from setuptools.command.build_ext import build_ext
import pybind11
from setuptools.command.sdist import sdist
from wheel._bdist_wheel import bdist_wheel
from auditwheel.repair import repair_wheel


def get_lib_name():
    if platform.system() == "Darwin":
        return "libllama-embedder.dylib"
    elif platform.system() == "Linux":
        return "libllama-embedder.so"
    elif platform.system() == "Windows":
        return ["llama-embedder.dll", "llama-embedder.lib"]
    else:
        raise OSError(f"Unsupported operating system: {platform.system()}")


def find_and_copy_win_shared_lib(build_dir, files_to_find: List[str], target_dir: str):
    for root, dirs, files in os.walk(build_dir):
        for file in files_to_find:
            if file in files:
                source_file = os.path.join(root, file)
                target_file = os.path.join(target_dir, file)
                shutil.copy2(source_file, target_file)
                print(f"Copied {source_file} to {target_file}")


class CustomBuildExt(build_ext):
    def run(self):
        print("doing CustomBuildExt")
        if platform.system() != "Windows":
            shared_lib_path = os.path.join('build', get_lib_name())
            print(f"Looking for shared library at: {shared_lib_path}")

            if not os.path.exists(shared_lib_path):
                raise FileNotFoundError(f"Shared library not found at {shared_lib_path}")
            shutil.copy2(shared_lib_path, self.build_lib)
        elif platform.system() == "Windows":
            find_and_copy_win_shared_lib("build", get_lib_name(), self.build_lib)
        # dest_path = os.path.join(self.build_lib)
        # os.makedirs(dest_path, exist_ok=True)
        # self.copy_file(shared_lib_path, self.build_lib)
        build_ext.run(self)
        if platform.system() == "Darwin": # post-processing of MacOS lib to fix linking
            extension_path = self.get_ext_fullpath('llama_embedder')
            cmd = ['install_name_tool', '-change', f'@rpath/{get_lib_name()}', f'@loader_path/{get_lib_name()}',
                   extension_path]
            subprocess.check_call(cmd)

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = []
        extra_link_args = []
        if ct == "unix":
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
            opts.append("-std=c++11")
            extra_link_args.append("-L" + self.build_lib)
            if platform.system() == "Darwin":
                opts.extend(["-stdlib=libc++", "-mmacosx-version-min=10.9"])
                extra_link_args.append("-Wl,-rpath,@loader_path/")
            elif platform.system() == "Linux":
                opts.append("-fvisibility=hidden")
                extra_link_args.append("-Wl,-rpath,$ORIGIN")
        elif ct == "msvc":
            extra_link_args.append("/LIBPATH:" + self.build_lib)
            opts.append(f'/DVERSION_INFO=\\"{self.distribution.get_version()}\\"')

        for ext in self.extensions:
            ext.extra_compile_args = opts
            if ext.extra_link_args is None:
                ext.extra_link_args = []
            ext.extra_link_args.extend(extra_link_args)
        build_ext.build_extensions(self)

ext_modules = [
    Extension(
        "llama_embedder",
        ["bindings/python/bindings.cpp"],
        include_dirs=[
            pybind11.get_include(),
            "bindings/python/",
            "src",  # Adjust this path to point to your C++ headers
        ],
        # library_dirs=["."],  # Adjust this path to point to your built libraries
        libraries=["llama-embedder"],
        library_dirs=[os.getcwd()],  # Add current working directory
        language="c++",
        extra_link_args=[
            "-L.",
            "-lllama-embedder",
        ],
    ),
]


class CustomSdist(sdist):
    """
    Here we create the source distribution.
    """

    def make_release_tree(self, base_dir, files):
        sdist.make_release_tree(self, base_dir, files)
        shutil.copytree("src", os.path.join(base_dir, "src"), dirs_exist_ok=True)
        shutil.copy2("CMakeLists.txt", base_dir)
        shutil.copy2("CMakePresets.json", base_dir)
        shutil.copy2("LICENSE.md", base_dir)
        os.makedirs(os.path.join(base_dir, "vendor/llama.cpp"), exist_ok=True)
        shutil.copytree("vendor/llama.cpp", os.path.join(base_dir, "vendor/llama.cpp"), dirs_exist_ok=True)


class CustomBdistWheel(bdist_wheel):
    """
    Here we create the release tree by adding the necessary build deps such as the shared lib and the src or header files
    """

    def run(self):
        wheel_dir = self.dist_dir
        package_dir = os.path.join(wheel_dir, 'llama_embedder')
        os.makedirs(package_dir, exist_ok=True)
        if platform.system() != "Windows":
            _shared_lib = os.path.join("build", get_lib_name())
            if not os.path.exists(_shared_lib):
                raise FileNotFoundError(f"Shared library not found at {_shared_lib}")
            dest = os.path.join(package_dir, os.path.basename(_shared_lib))
            shutil.copy2(_shared_lib, Path(dest).parent)
        elif platform.system() == "Windows":
            find_and_copy_win_shared_lib("build", get_lib_name(), str(Path(package_dir).parent))

        dest_src_path = os.path.join(self.dist_dir, "src")
        shutil.copytree("src", dest_src_path, dirs_exist_ok=True)
        shutil.copy2("LICENSE.md", self.dist_dir)
        os.makedirs(os.path.join(self.dist_dir, "vendor/llama.cpp"), exist_ok=True)
        shutil.copy2("vendor/llama.cpp/LICENSE", os.path.join(self.dist_dir, "vendor/llama.cpp/LICENSE"))
        # Call the standard run method
        super().run()


setup(
    package_data={"llama_embedder": [f"{get_lib_name()}"] if platform.system() != "Windows" else get_lib_name()},
    include_package_data=True,
    zip_safe=False,
    ext_modules=ext_modules,
    cmdclass={"build_ext": CustomBuildExt, "sdist": CustomSdist, "bdist_wheel": CustomBdistWheel, },
)
