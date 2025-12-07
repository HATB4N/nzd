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
- idk

## 한거
- 일단 곱셈 되긴 함
- 세이브 로드 일단은 됨

## 언젠간 할거
- matrix 클래스의 thread 수명은 unique_ptr로 관리. 생성 오버헤드 줄이기
- operations를 unit화 후 graph engine만들고 dependencies에 맞게 펼쳐서 쓰레드 분배하기
- 레이어도 인터페이스화