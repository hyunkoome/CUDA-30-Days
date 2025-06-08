cmake_minimum_required(VERSION 3.18)
project(matrix_add_all LANGUAGES CXX CUDA)

# 표준 설정
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_ARCHITECTURES 89)  # RTX 4090
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# 경로 지정
include_directories(${PROJECT_SOURCE_DIR}/common)
include_directories(${PROJECT_SOURCE_DIR}/test_libtorch)

# ──────────────── LibTorch 설정 ────────────────
# 필요 시 아래 경로 수정
set(Torch_DIR "/home/hyunkoo/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)

# 공통 연산 코드 (Python.h 포함되지 않음)
set(SRC_COMMON
    common/kernel_matrix_add.cu
    common/matrix_add_impl.cpp
    common/matrix_add_impl.h
)

# ──────────────── CUDA 전용 테스트 ────────────────
add_executable(test_runner
    test_cu/test_runner.cu
    ${SRC_COMMON}
)
# target_link_libraries(test_runner ${TORCH_LIBRARIES})
target_link_libraries(test_runner ${TORCH_LIBRARIES} m)
target_include_directories(test_runner PRIVATE ${TORCH_INCLUDE_DIRS})
set_property(TARGET test_runner PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# ──────────────── C++ LibTorch 테스트 ────────────────
add_executable(test_libtorch
    test_libtorch/test_libtorch.cpp
    test_libtorch/wrapper_libtorch.cpp
    ${SRC_COMMON}
)
#target_link_libraries(test_libtorch ${TORCH_LIBRARIES})
target_link_libraries(test_libtorch ${TORCH_LIBRARIES} m)
target_include_directories(test_libtorch PRIVATE ${TORCH_INCLUDE_DIRS})

# 선택: C++ ABI 설정 (PyTorch와 컴파일러 ABI mismatch 방지)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)
