pypi: python-dist
	twine upload dist/*
.PHONY: python-dist
python-dist: lib
	cd bindings/python && rm -rf dist/*
	cd bindings/python && pip install build
	cd bindings/python && python3 -m build

python-cidist-local:
	rm -rf dist/*
	rm -rf build/lib.*
	rm -rf build/temp.*
	pip install cibuildwheel==2.19.1 auditwheel
	CIBW_BEFORE_BUILD="make lib" \
	CIBW_SKIP="pp* *musllinux*" \
	CIBW_ARCHS_MACOS="x86_64" \
	CIBW_PROJECT_REQUIRES_PYTHON=">=3.8,<3.9" \
	CI=1 \
	python -m cibuildwheel --output-dir dist

python-cidist:
	rm -rf dist/*
	rm -rf build/
	pip install cibuildwheel==2.19.1 auditwheel
	python -m cibuildwheel --output-dir dist

python-test: python-dist
	cd bindings/python && pip install pytest
	cd bindings/python && pip install --force-reinstall dist/*.whl
	cd bindings/python/tests && pytest

python-clean:
	rm -rf *.egg-info build dist

ARCH := "${_PYTHON_HOST_PLATFORM}"
IS_X86 = false
ifeq ($(findstring x86_64,$(ARCH)),x86_64)
    IS_X86 = true
endif

CMAKE_ARCH_FLAG = ""
# Add architecture flag based on the target platform
ifeq ($(findstring win_arm64,$(ARCH)),win_arm64)
    CMAKE_ARCH_FLAG = "-A ARM64"
else ifeq ($(findstring win_amd64,$(ARCH)),win_amd64)
    CMAKE_ARCH_FLAG = "-A x64"
endif

# Function to detect target architecture
define detect_arch
$(shell \
  if [ -n "$$CIBW_ARCHS" ]; then \
    case "$$CIBW_ARCHS" in \
      *arm64*|*aarch64*) echo "arm64" ;; \
      *x86_64*|*AMD64*) echo "x86_64" ;; \
      *) echo "unknown" ;; \
    esac; \
  elif [ -n "$$_PYTHON_HOST_PLATFORM" ]; then \
    case "$$_PYTHON_HOST_PLATFORM" in \
      *arm64*|*aarch64*) echo "arm64" ;; \
      *x86_64*) echo "x86_64" ;; \
      *) echo "unknown" ;; \
    esac; \
  else \
    case "$$(uname -m)" in \
      arm64|aarch64) echo "arm64" ;; \
      x86_64|amd64) echo "x86_64" ;; \
      *) echo "unknown" ;; \
    esac; \
  fi \
)
endef

TARGET_ARCH := $(call detect_arch)

lib:
	@echo "Building for $(TARGET_ARCH)"
	rm -rf build && mkdir build
	@if [ "$(IS_X86)" = "true" ]; then \
		arch -x86_64 /bin/bash -c "cd build && cmake ${CMAKE_FLAGS} ${CMAKE_ARCH_FLAG} -DGGML_NATIVE=OFF -DLLAMA_BUILD_SERVER=ON -DGGML_RPC=ON -DGGML_AVX=OFF -DGGML_AVX2=OFF -DGGML_FMA=OFF -DBUILD_SHARED_LIBS=OFF .. && cmake --build . --config Release"; \
	else \
		cd build && cmake ${CMAKE_FLAGS} ${CMAKE_ARCH_FLAG} -DGGML_NATIVE=OFF -DLLAMA_BUILD_SERVER=ON -DGGML_RPC=ON -DGGML_AVX=OFF -DGGML_AVX2=OFF -DGGML_FMA=OFF -DBUILD_SHARED_LIBS=OFF .. && cmake --build . --config Release; \
	fi
#-j $(sysctl -n hw.logicalcpu)