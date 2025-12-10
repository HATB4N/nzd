# NaZakDeep

## 뭐하는거?
STL만 써서 신경망 만들기(cpp20 req)


## 사용
### 기능
- nothing

### 빌드
```sh
make
```

### 실행
```sh
./NZD
```

## 진행도
0.342532% / 100%

### 당장 할거
- back propagation -> mnist 학습
- mini batch
- etc

### 한거
- 일단 곱셈 되긴 함
- 세이브 로드 일단은 됨

### 언젠간 할거
- matrix 클래스의 thread 수명은 unique_ptr로 관리. 생성 오버헤드 줄이기 => Threadpool 구성함
- operations를 unit화 후 graph engine만들고 dependencies에 맞게 펼쳐서 쓰레드 분배하기
- 레이어도 인터페이스화

#### 언젠간 할거 메모
- control plane과 execution plane을 구분한다.
- control plane에서는 추상화된 레이어만을 바라보며 유저의 설정에 따라 명령을 내린다.
    - 실제 각 레이어에서는 type에 맞는 sequential execution plan을 구성한다.
    - 중간 계층(ICompiler? Optimizer? idk)에서는 위에서 내려온 plan이 무엇이든 이에 대한 최적화를 진행하여 execution plane으로 내려보낸다. (내려오는게 affine -> activate -> ... backpropa. -> update형식이든, conv. -> relu -> pooling -> ...이든, QKV든 뭐든 공통된 형식으로 기술된 execution들에 대한 최적화만)
- execution plane에서는 내려온 plan에 대한 실행을 담당한다. (나중에 aarch64, amd64구분해서 SIMD를 직접 쓰든 뭘 하든 여기서 동일 interface를 보장하는 여러 연산 모듈 중 필요한 걸 선택할수 있게)

## 참고
- (언젠가 무조건 바뀔)파라미터 저장 방식
    - `[NZD(magicbyte)][metadata len][metadata(about how to read the data)]` 하나 넣고(레이어들에 대한 정보)
    - `[metadata len][metadata][data]`단위를 하위 함수에서 순차적으로 write함(하나의 레이어에 대한 정보).

## 진짜 참고
- 프로젝트 이름 지어준 사람: [@zenisa1](https://github.com/zenisa1)
