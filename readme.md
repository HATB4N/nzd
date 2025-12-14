# NaZakDeep
C++20 based deep learning framework(not really)

## 할 거
### 1단계: 기본 신경망 검증
- [x] Fully Connected (Dense) Layers
- [x] Activation Functions(e.g. ReLU, Softmax)
- [ ] Optimizers: SGD, Adam(WIP)
- [ ] Loss Functions: Cross-Entropy(WIP)
- [x] Weight Initializers: Xavier, He
- [x] MNIST 데이터셋 학습 성공 (ref: [251213](./notes/memo/251213.md))

### 2단계: 기능 확장 & 일반화(수립중)
- [ ] **정규화 (Regularization)**
    - [ ] Batch Normalization
    - [ ] Dropout
- [ ] **코어 아키텍처 리팩토링** (ref: [251213](./notes/memo/251213.md))
    - [ ] 메모리 풀 (Memory Pool)
    - [ ] 매트릭스 뷰 (Matrix View)
    - [ ] 연산자 백엔드 추상화 (for AVX/NEON)
- [ ] **레이어 추가(일반화 검증용)**
    - [ ] Convolutional Layers

### 3단계: 컴파일러 기반 최적화(수립중)
- [ ] 그래프 기반 실행 엔진 구축
- [ ] 그래프 최적화 컴파일러 구축
- [ ] 설정 / 최적화 / 실행 플레인 분리
- [ ] 하드웨어 의존성 최적화 구현

## 빌드 및 실행 (Build and Run)
### 요구 사항
* idk

### 빌드
```sh
make
```

### 실행
```sh
./NZD
```

## 대충 있어보이는 크레딧
* 프로젝트 이름 제안: [@zenisa1](https://github.com/zenisa1)
* References: [references](./notes/references.md)