// test_runner.cpp

/*
1) extern "C"
    :   C++ 링커가 CUDA 함수를 제대로 연결하도록 이름 변경 방지

2) a[N], b[N], c[N]
    :   CPU 메모리에 벡터를 저장

3) vec_add_launcher()
    :   GPU에서 벡터 덧셈을 수행하는 CUDA 함수 호출

4) std::cout
    :   결과 확인을 위한 출력
*/

#include <iostream>  // C++ 표준 입출력 스트림 헤더

// CUDA에서 구현된 함수는 C++과 연동을 위해 extern "C" 선언 필요
// CUDA 파일에서 정의된 vec_add_launcher() 함수를 가져옴
extern "C" void vec_add_launcher(float* a, float* b, float* c, int N);

int main() {
    const int N = 100;         // 벡터 길이 정의
    float a[N], b[N], c[N];    // 입력 벡터 a, b와 결과 벡터 c 선언

    // 입력 데이터 초기화
    // a = [0, 1, 2, ..., 99]
    // b = [0, 2, 4, ..., 198]
    for (int i = 0; i < N; i++) {
        a[i] = i;
        b[i] = 2 * i;
    }

    // CUDA 런처 함수 호출 (GPU에서 벡터 덧셈 수행)
    vec_add_launcher(a, b, c, N);

    // 결과 일부 출력 (디버깅 용도)
    for (int i = 0; i < 5; i++) {
        std::cout << a[i] << " + " << b[i] << " = " << c[i] << std::endl;
        // 예: 0 + 0 = 0, 1 + 2 = 3, ...
    }

    return 0;  // 프로그램 종료
}
