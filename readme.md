# NaZakDeep

## 뭐하는거?
STL만 써서 신경망 만들기(cpp23 req)

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

## 할거
- back propagation
- mini batch
- etc
- idk

## 참고
- 메모리 순차 접근을 위해 R = WX+B에서, X를 X^T로 저장함. (R도 동일)
- minibatch를 col 단위로 구분(전치)

```
16384 * 4096 * 2048, 14Threads
전치 전: 7-80sec (비순차)
전치 후: 9.5sec (순차)
```


- 추상화 개층을 추가해서 실제로는 일반 행렬 다루듯이 가능하게.
- 미니배치를 위한 구성이니까, WX말고, XW로 곱하는 경우는 고려하지 않아도 될듯...?