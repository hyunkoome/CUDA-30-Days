// shared_memory_mixed_precision_kernel.cu

#include <cuda_fp16.h>     // ✅ half 자료형(__half) 사용을 위한 헤더
#include <cuda_runtime.h>  // ✅ CUDA 런타임 API 헤더

#define THREADS 256  // ✅ 블록 당 스레드 수 (공유 메모리 크기 및 병렬 처리 단위)

extern "C"
__global__ void dotProductHalf(const __half* a, const __half* b, float* result, int size) {
    // ✅ block 내 모든 스레드가 공유하는 캐시 (합산을 위한 중간 버퍼)
    __shared__ float cache[THREADS];

    // ✅ 전체 쓰레드에서 고유 인덱스 계산 (1D 격자 기반)
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // ✅ 공유 메모리 내에서 나의 위치 인덱스
    int cacheIdx = threadIdx.x;

    // ✅ 임시 합계 변수 (float로 누적)
    float temp = 0.0f;

    // ✅ 입력 배열 전체를 stride 방식으로 순회
    while (tid < size) {
        // ⬇️ half 자료형을 float으로 변환하여 곱셈 수행
        temp += __half2float(a[tid]) * __half2float(b[tid]);

        // ⬇️ 다음 인덱스는 블록 전체를 건너뛴 위치로 이동 (grid stride loop)
        tid += blockDim.x * gridDim.x;
    }

    // ✅ 내 결과를 공유 메모리 캐시에 저장
    cache[cacheIdx] = temp;
    __syncthreads();  // ⬅️ 모든 스레드가 캐시에 쓰기 완료할 때까지 대기

    // ✅ 병렬 Reduction: 1블록 내 합산을 반복적으로 반으로 줄여나감
    for (int i = blockDim.x / 2; i > 0; i >>= 1) {
        // ⬇️ 절반 크기만큼 더 작은 인덱스에서 큰 인덱스의 값을 더함
        if (cacheIdx < i) {
            cache[cacheIdx] += cache[cacheIdx + i];
        }
        __syncthreads();  // ⬅️ 덧셈 완료 후 다시 동기화
    }

    // ✅ 블록 내 첫 번째 스레드가 결과를 전역 메모리에 원자적으로 더함
    if (cacheIdx == 0) {
        atomicAdd(result, cache[0]);
    }
}
