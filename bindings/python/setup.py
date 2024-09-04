import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

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
        # return ["llama-embedder.dll","llama-embedder.lib"]
        return "llama-embedder.dll"
    else:
        raise OSError(f"Unsupported operating system: {platform.system()}")


class CustomBuildExt(build_ext):
    def run(self):
        print("doing CustomBuildExt")
        shared_lib_path = os.path.join('build', get_lib_name())
        print(f"Looking for shared library at: {shared_lib_path}")

        if not os.path.exists(shared_lib_path):
            raise FileNotFoundError(f"Shared library not found at {shared_lib_path}")

        # dest_path = os.path.join(self.build_lib)
        # os.makedirs(dest_path, exist_ok=True)
        shutil.copy2(shared_lib_path, self.build_lib)
        print(os.listdir(self.build_lib))
        print(os.path.abspath(self.build_lib))
        # self.copy_file(shared_lib_path, self.build_lib)
        build_ext.run(self)
        if platform.system() == "Darwin":
            extension_path = self.get_ext_fullpath('llama_embedder')
            cmd = ['install_name_tool', '-change', f'@rpath/{get_lib_name()}', f'@loader_path/{get_lib_name()}', extension_path]
            subprocess.check_call(cmd)

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = []
        extra_link_args=[]
        if ct == "unix":
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
            opts.append("-std=c++11")
            extra_link_args.append("-L"+self.build_lib)
            if platform.system() == "Darwin":
                opts.extend(["-stdlib=libc++", "-mmacosx-version-min=10.9"])
                extra_link_args.append("-Wl,-rpath,@loader_path/")
            elif platform.system() == "Linux":
                opts.append("-fvisibility=hidden")
                extra_link_args.append("-Wl,-rpath,$ORIGIN")
        elif ct == "msvc":
            opts.append(f'/DVERSION_INFO=\\"{self.distribution.get_version()}\\"')

        for ext in self.extensions:
            ext.extra_compile_args = opts
            if ext.extra_link_args is None:
                ext.extra_link_args = []
            ext.extra_link_args.extend(extra_link_args)
        build_ext.build_extensions(self)

extra_link_args = [
    "-L.",
    "-lllama-embedder",
    ]

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
        extra_link_args=extra_link_args,
    ),
]


class CustomSdist(sdist):
    """
    Here we create the release tree by adding the necessary build deps such as the shared lib and the src or header files
    """

    def make_release_tree(self, base_dir, files):
        sdist.make_release_tree(self, base_dir, files)
        # Copy shared library to the base dir of the source distribution
        dest = os.path.join(base_dir, get_lib_name())
        shutil.copy2(get_lib_name(), dest)
        dest_src_path = os.path.join(base_dir, "src")
        shutil.copytree("src", dest_src_path, dirs_exist_ok=True)
        shutil.copy2("LICENSE.md", base_dir)
        os.makedirs(os.path.join(base_dir, "vendor/llama.cpp"), exist_ok=True)
        shutil.copy2("vendor/llama.cpp/LICENSE", os.path.join(base_dir, "vendor/llama.cpp/LICENSE"))


class CustomBdistWheel(bdist_wheel):
    """
    Here we create the release tree by adding the necessary build deps such as the shared lib and the src or header files
    """

    def run(self):
        _shared_lib = os.path.join("build", get_lib_name())
        print("CustomBdistWheel",os.path.abspath(self.dist_dir))
        if not os.path.exists(_shared_lib):
            raise FileNotFoundError(f"Shared library not found at {_shared_lib}")

        # Create the llama_embedder directory in the wheel
        wheel_dir = self.dist_dir
        package_dir = os.path.join(wheel_dir, 'llama_embedder')
        os.makedirs(package_dir, exist_ok=True)

        # Copy the shared library to the package directory
        dest = os.path.join(package_dir, os.path.basename(_shared_lib))
        shutil.copy2(_shared_lib, Path(dest).parent)
        print(os.listdir(Path(dest).parent))

        dest_src_path = os.path.join(self.dist_dir, "src")
        shutil.copytree("src", dest_src_path, dirs_exist_ok=True)
        shutil.copy2("LICENSE.md", self.dist_dir)
        os.makedirs(os.path.join(self.dist_dir, "vendor/llama.cpp"), exist_ok=True)
        shutil.copy2("vendor/llama.cpp/LICENSE", os.path.join(self.dist_dir, "vendor/llama.cpp/LICENSE"))
        # Call the standard run method
        super().run()


setup(
    package_data={"llama_embedder": [f"{get_lib_name()}"]},
    include_package_data=True,
    zip_safe=False,
    ext_modules=ext_modules,
    cmdclass={"build_ext": CustomBuildExt,"sdist": CustomSdist,"bdist_wheel": CustomBdistWheel,},
)
