pypi: python-dist
	twine upload dist/*
.PHONY: python-dist
python-dist: lib
	cd bindings/python && rm -rf dist/*
	cd bindings/python && pip install build
	cd bindings/python && python3 -m build

python-test: python-dist
	cd bindings/python && pip install pytest
	cd bindings/python && pip install --force-reinstall dist/*.whl
	cd bindings/python/tests && pytest

python-clean:
	rm -rf *.egg-info build dist tmp var tests/__pycache__ hnswlib.cpython*.so

lib:
	rm -rf build && mkdir build
	cd build && cmake -DLLAMA_FATAL_WARNINGS=ON -DGGML_METAL_EMBED_LIBRARY=ON -DLLAMA_CURL=ON -DGGML_RPC=ON -DBUILD_SHARED_LIBS=ON .. && cmake --build . --config Release -j $(sysctl -n hw.logicalcpu)