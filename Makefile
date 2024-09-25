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
	CIBW_BEFORE_BUILD="make lib-test" \
	CIBW_SKIP="pp* *musllinux*" \
	CIBW_ARCHS_MACOS="arm64" \
  	CIBW_ARCHS_WINDOWS="AMD64" \
  	CIBW_ARCHS_LINUX="x86_64 aarch64" \
	CIBW_PROJECT_REQUIRES_PYTHON=">=3.9,<3.10" \
  	CIBW_TEST_REQUIRES="pytest>=6.0.0 huggingface_hub" \
  	CIBW_TEST_COMMAND="python -m pytest {project}/bindings/python/tests/test" \
	CI=1 \
	python -m cibuildwheel --output-dir dist

python-cidist:
	rm -rf dist/*
	rm -rf build/
	pip install cibuildwheel==2.19.1 auditwheel
	python -m cibuildwheel --output-dir dist

python-sdist:
	python3 -m build --sdist
python-test: python-dist
	cd bindings/python && pip install pytest
	cd bindings/python && pip install --force-reinstall dist/*.whl
	cd bindings/python/tests && pytest

python-clean:
	rm -rf *.egg-info build dist

go-test: lib-test
	cd bindings/go && go test -v ./...

.PHONY: lint
go-lint:
	cd bindings/go && golangci-lint run

.PHONY: lint-fix
go-lint-fix:
	cd bindings/go && golangci-lint run --fix ./...

ARCH := "${_PYTHON_HOST_PLATFORM}"
IS_X86 = false
ifeq ($(findstring x86_64,$(ARCH)),x86_64)
    IS_X86 = true
endif

lib:
	rm -rf build && mkdir build
	@if [ "$(IS_X86)" = "true" ]; then \
		arch -x86_64 /bin/bash -c "cd build && cmake ${CMAKE_FLAGS} -DGGML_NATIVE=OFF -DLLAMA_BUILD_SERVER=ON -DGGML_RPC=ON -DGGML_AVX=OFF -DGGML_AVX2=OFF -DGGML_FMA=OFF -DBUILD_SHARED_LIBS=OFF .. && cmake --build . --config Release"; \
	else \
		cd build && cmake ${CMAKE_FLAGS} -DGGML_NATIVE=OFF -DLLAMA_BUILD_SERVER=ON -DGGML_RPC=ON -DGGML_AVX=OFF -DGGML_AVX2=OFF -DGGML_FMA=OFF -DBUILD_SHARED_LIBS=OFF .. && cmake --build . --config Release ${CMAKE_BUILD_FLAGS}; \
	fi

lib-test: lib
	pip install huggingface_hub
	huggingface-cli download ChristianAzinn/snowflake-arctic-embed-s-gguf --include=snowflake-arctic-embed-s-f16.GGUF --local-dir build/snowflake-arctic-embed-s
	cd build && ctest -V
#-j $(sysctl -n hw.logicalcpu)

lib-static:
	rm -rf build && mkdir build
	@if [ "$(IS_X86)" = "true" ]; then \
		arch -x86_64 /bin/bash -c "cd build && cmake ${CMAKE_FLAGS} -DLLAMA_EMBEDDER_BUILD_STATIC=ON -DGGML_NATIVE=OFF -DLLAMA_BUILD_SERVER=ON -DGGML_RPC=ON -DGGML_AVX=OFF -DGGML_AVX2=OFF -DGGML_FMA=OFF -DBUILD_SHARED_LIBS=OFF .. && cmake --build . --config Release"; \
	else \
		cd build && cmake ${CMAKE_FLAGS} -DLLAMA_EMBEDDER_BUILD_STATIC=ON -DGGML_NATIVE=OFF -DLLAMA_BUILD_SERVER=OFF -DGGML_RPC=OFF -DGGML_AVX=OFF -DGGML_AVX2=OFF -DGGML_FMA=OFF -DBUILD_SHARED_LIBS=OFF .. && cmake --build . --config Release ${CMAKE_BUILD_FLAGS}; \
	fi

lib-static-test: lib-static
	pip install huggingface_hub
	huggingface-cli download ChristianAzinn/snowflake-arctic-embed-s-gguf --include=snowflake-arctic-embed-s-f16.GGUF --local-dir build/snowflake-arctic-embed-s
	cd build && ctest -V