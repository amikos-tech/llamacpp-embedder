pypi: python-dist
	twine upload dist/*
.PHONY: python-dist
python-dist: lib
	cd bindings/python && rm -rf dist/*
	cd bindings/python && pip install build
	cd bindings/python && python3 -m build

python-cidist: lib
	rm -rf bindings/python/dist/*
	pip install cibuildwheel==2.19.1 auditwheel
	export CIBW_SKIP="pp* *musllinux*"
	export CI=1
	python -m cibuildwheel --output-dir dist

python-test: python-dist
	cd bindings/python && pip install pytest
	cd bindings/python && pip install --force-reinstall dist/*.whl
	cd bindings/python/tests && pytest

python-clean:
	rm -rf *.egg-info build dist tmp var tests/__pycache__ hnswlib.cpython*.so

lib:
	rm -rf build && mkdir build
	cd build && cmake ${CMAKE_FLAGS} -DGGML_NATIVE=OFF -DLLAMA_BUILD_SERVER=ON -DGGML_RPC=ON -DGGML_AVX=OFF -DGGML_AVX2=OFF -DGGML_FMA=OFF -DBUILD_SHARED_LIBS=OFF .. && cmake --build . --config Release
#-j $(sysctl -n hw.logicalcpu)