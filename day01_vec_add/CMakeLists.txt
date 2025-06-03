cmake_minimum_required(VERSION 3.18)
project(vec_add LANGUAGES CXX CUDA)

# ─────────────────────────────────────────────
# 표준 설정
# ─────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# 디버깅과 경고 옵션
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo -Xcompiler -Wall")

# RTX 4090 아키텍처 (Ada Lovelace)
set(CMAKE_CUDA_ARCHITECTURES 89)

# 디폴트 빌드 타입
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# 실행파일 출력 위치 설정
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# CLion 자동완성용 compile_commands.json 생성
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ─────────────────────────────────────────────
# 타겟 추가
# ─────────────────────────────────────────────
add_executable(vec_add_runner
    test_runner.cpp
    vec_add_kernel.cu
)

# 타겟 속성 설정 (선택 사항)
set_target_properties(vec_add_runner PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
