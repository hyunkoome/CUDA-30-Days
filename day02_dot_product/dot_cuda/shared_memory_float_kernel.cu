// shared_memory_float_kernel.cu

#include <cuda_runtime.h>

#define THREADS 256  // 하나의 블록당 쓰레드 수

// 외부에서 호출 가능한 CUDA 커널 함수
extern "C"
__global__ void dotProductShared(const float* a, const float* b, float* result, int size) {
    // 블록 내 공유 메모리 할당 (동일 블록의 쓰레드 간 데이터 공유용)
    __shared__ float cache[THREADS];

    // 전체 글로벌 인덱스 계산: threadIdx + blockIdx * blockDim
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // 공유 메모리 내 인덱스 (현재 쓰레드의 위치)
    int cacheIdx = threadIdx.x;

    float temp = 0.0f;
    /*
    blockDim.x: 한 블록 안의 쓰레드 수
    gridDim.x: 전체 블록 수
    따라서 blockDim.x * gridDim.x: 전체 쓰레드 총합

    ✅ Strided loop: 한 쓰레드가 처리할 인덱스를 일정 간격으로 점프하며 반복 처리
    예: tid = 0 → 512 → 1024 ...  (blockDim.x * gridDim.x = 전체 쓰레드 수)
    */
    while (tid < size) {
        temp += a[tid] * b[tid];  // 벡터 요소 곱셈 누적
        tid += blockDim.x * gridDim.x;  // 다음 루프에서 자기 차례 인덱스 점프
    }

    // 공유 메모리에 각 쓰레드의 누적 결과 저장
    cache[cacheIdx] = temp;

    // 모든 쓰레드가 공유 메모리 작성을 마칠 때까지 동기화
    __syncthreads();

    /*
    ✅ 병렬 reduction 수행: 공유 메모리에서 값을 절반씩 합치며 줄여나감

    cache[0] + cache[1] + cache[2] + ... + cache[255]
    → 이 전체 합을 cache[0] 하나에 저장하는 게 목적.

    i값	역할
    128	0~127번 쓰레드가 cache[i] += cache[i+128] 수행
    64	0~63번 쓰레드가 cache[i] += cache[i+64] 수행
    32	0~31번   쓰레드가 cache[i] += cache[i+32] 수행
    ...	...
    1	0번 쓰레드가 cache[0] += cache[1] 수행

    ✅ 시각화 예 (8개 쓰레드 예시)
    초기 상태 (temp 계산 후):
    cache = [1, 2, 3, 4, 5, 6, 7, 8]
    1단계: i = 4
    cache[0] += cache[4] → 1+5 = 6
    cache[1] += cache[5] → 2+6 = 8
    cache[2] += cache[6] → 3+7 = 10
    cache[3] += cache[7] → 4+8 = 12
    → cache = [6, 8, 10, 12, 5, 6, 7, 8]

    2단계: i = 2
    cache[0] += cache[2] → 6+10 = 16
    cache[1] += cache[3] → 8+12 = 20
    → cache = [16, 20, 10, 12, ...]

    3단계: i = 1
    cache[0] += cache[1] → 16+20 = 36
    결과: cache[0] == 총합 🎉

    */
    for (int i = blockDim.x / 2; i > 0; i >>= 1) {
        if (cacheIdx < i) {
            cache[cacheIdx] += cache[cacheIdx + i];
        }
        __syncthreads();  // 매 단계마다 쓰레드 동기화
    }

    /*
    CUDA에서 블록 수는 보통 아래와 같이 자동 계산되는데, test_runner.cu에서 아래처럼 쓰셨죠:
    int blocks = (N + THREADS - 1) / THREADS;
    N = 1 << 20 → 1,048,576 (총 데이터 개수)
    THREADS = 256 → 한 블록당 쓰레드 수
    따라서:
        blocks = (1048576 + 256 - 1) / 256 = 4096
    ✅ 결론
        총 4096개의 블록이 생성됩니다.
    그리고 각 블록의 threadIdx.x == 0 쓰레드만 cache[0]을 전역 결과 result에 atomicAdd() 합니다.

    즉, 총 4096개의 atomicAdd() 연산이 실행되는 셈입니다.
    */

    // 블록 내 첫 번째 쓰레드가 최종 결과를 전역 메모리에 atomic 방식으로 저장
    if (cacheIdx == 0) // 블록 내 0번 쓰레드만 실행 (block당 1개)
    {
        // cache[0]: 블록 전체 쓰레드의 내적 누적합
        // atomicAdd: 여러 블록이 동시에 접근해도 안전하게 전역 합산
        atomicAdd(result, cache[0]);  // 🔥 단 하나의 쓰레드만 result 에 더함
    }

    /*
    1. 각 쓰레드가 자신의 역할만큼 내적 계산 (a[tid] * b[tid])
    2. 공유 메모리 (cache)에 각 쓰레드 결과 저장
    3. __syncthreads() → 병렬 reduction 수행
    4. 최종적으로 cache[0]에 블록 전체 합계가 모임
    5. cacheIdx == 0인 쓰레드만 → result에 atomicAdd()
    */
}
