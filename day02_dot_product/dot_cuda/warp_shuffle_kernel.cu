// warp_shuffle_kernel.cu

#include <cuda_runtime.h>

/*
이 코드는 warp shuffle을 이용한 reduction 구현이며, shared memory를 전혀 사용하지 않음
CUDA의 warp-level intrinsic 함수 __shfl_down_sync()을 사용하여 동일 warp 내에서만 병렬 reduction을 수행함

🧠 요약
항목	            설명
shared memory	❌ 사용 안 함
reduction 방식	와프 warp shuffle
속도	            매우 빠름 (warp 내부 통신은 레지스터 기반)
제한점	        32개 thread(warp)까지만 가능 → warp 간 통신은 직접 구현 필요
*/

/*
🧠 0xffffffff는 32비트 unsigned 정수의 모든 비트를 1로 설정한 값
    16진수로는 f가 4비트(1111)이므로 8개 쓰면 32비트가 전부 1인 상태:
    0xffffffff  ==  11111111 11111111 11111111 11111111 (32비트)

🧠 CUDA에서 0xffffffff의 의미
    __shfl_down_sync(mask, var, offset) 함수에서 mask는 어떤 thread들이 이 warp-level shuffle 연산에 참여할지 지정하는 비트 마스크 임
        0xffffffff는 warp 내의 모든 32개 thread가 다 참여한다는 뜻임
        즉, "현재 warp의 모든 thread가 유효하다(valid)"는 의미로 사용됨

    __shfl_down_sync(0xffffffff, val, 1);
        → 같은 warp 안에서 모든 thread들이 참여해서, val을 한 칸씩 아래 thread로 이동하며 연산

🧠 참고: 마스크를 다르게 쓰는 경우?
    만약 일부 thread만 유효하다면,
    예를 들어 0x0000ffff는 하위 16개 thread만,
    0xffff0000는 상위 16개 thread만을
    대상으로 할 수도 있음
*/

/// ✅ warp 단위로 병렬 reduction 수행하는 함수
/// 한 warp(32 threads) 내에서만 작동하며, 공유 메모리 없이 빠르게 동작함

__inline__ __device__ float warpReduceSum(float val) {
    // offset: 16 → 8 → 4 → 2 → 1 순서로 절반씩 줄이면서 더함
    for (int offset = 16; offset > 0; offset /= 2)
        // __shfl_down_sync(mask, var, offset):
        //   현재 thread의 var 값을 offset만큼 낮은 thread에서 받아와 더함
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

/// ✅ 전체 벡터 내적 (dot product) 커널 함수
/// 공유 메모리 대신 warp shuffle을 이용하여 고속 reduction 수행
extern "C"
__global__ void dotProductWarp(const float* a, const float* b, float* result, int size) {
    float sum = 0.0f;

    // 전역 thread 인덱스 계산
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // 1D 그리드로 전체 배열 순회하며 곱셈 누적
    while (tid < size) {
        sum += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }

    // ✅ 같은 warp 내에서만 병렬로 합산 (shared memory 불필요)
    sum = warpReduceSum(sum);

    // ✅ warp의 첫 번째 thread만 전역 결과에 원자적 추가
    //   threadIdx.x & 31 == 0 : warp의 리더 thread 조건 (0, 32, 64, ...)
    if ((threadIdx.x & 31) == 0)
        atomicAdd(result, sum);
}
