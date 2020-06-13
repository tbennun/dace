import os
import dace.library

CONDA_HOME = "/home/orausch/.local/opt/miniconda3/envs/dace"

@dace.library.environment
class ONNXRuntime:
    cmake_minimum_version = None
    cmake_packages = []
    cmake_variables = {}
    cmake_includes = ["/home/orausch/.local/opt/miniconda3/envs/dace/lib/python3.7/site-packages", # pick up the onnx source headers
                      "/home/orausch/sources/onnx/build-no-ml" # pick up the compiled protobuf headers
                      ]
    cmake_libraries = [] #["/home/orausch/.local/opt/miniconda3/envs/dace/lib/libprotobuf.so"]
    cmake_compile_flags = []
    cmake_link_flags = []
    cmake_files = []

    headers = ["onnx/onnx_pb.h"]
    init_code = ""
    finalize_code = ""
