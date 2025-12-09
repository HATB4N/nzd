# NaZakDeep

## 뭐하는거?
STL만 써서 신경망 만들기(cpp20 req)

## 기능
- nothing

## 빌드
```sh
make
```

## 실행
```sh
./NZD
```

## 당장 할거
- back propagation -> mnist 학습
- mini batch
- etc
- init시점에서 env나 arg로 MAX_MEMORY 받게 해서 heap에 연속 memory pool만들고 Matrix_T는 저기서 메모리를 할당받음. static var(total memory usage)과 비교해서 우선순위에 맞춰 연속접근을 위해 캐시된 vector(_m_t or _m 중 최근에 접근하지 않은 것)에 대한 삭제 (LRU? idk)

## 한거
- 일단 곱셈 되긴 함
- 세이브 로드 일단은 됨

## 언젠간 할거
- matrix 클래스의 thread 수명은 unique_ptr로 관리. 생성 오버헤드 줄이기 => Threadpool 구성함
- operations를 unit화 후 graph engine만들고 dependencies에 맞게 펼쳐서 쓰레드 분배하기
- 레이어도 인터페이스화

## 참고
- 파라미터 저장 방식
    - `[NZD(magicbyte)][metadata len][metadata(about how to read the data)]` 하나 넣고(레이어들에 대한 정보)
    - `[metadata len][metadata][data]`단위를 하위 함수에서 순차적으로 write함(하나의 레이어에 대한 정보).
