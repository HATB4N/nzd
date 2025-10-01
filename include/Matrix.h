#ifndef MATRIX_H
#define MATRIX_H

#include <cstddef>
#include <stdfloat>
#include <vector>
#include <span>

// cpp23
using fp16 = std::float16_t;
using fp32 = std::float32_t;

class Matrix { // 생성자, 소멸자로 thread pool을 static 두는 것도 좋을 듯?
public:
    void setup(size_t m1_row, size_t m1_col, size_t m2_row, size_t m2_col);
    void add(std::vector<fp32> &m_t, const std::vector<fp16> &b);
    void multiply(const std::vector<fp16> &w, const std::vector<fp16> &xt, 
        std::vector<fp32> &rt);
private:
    void product(std::span<const fp16> w, std::span<const fp16> xt, 
        std::span<fp32> rt, size_t batch_p);
    
    unsigned int _threads = 8; // given by argv (fix)
    size_t _m1_row, _m1_col, _batch;

};

#endif

// 8t: 3.5 - 0.5 = 3sec?
// 14t: 3.3 - 0.5 = 2.8sec?
// 200t without limit: 21 - 0.5 = 20.5sec (코어 하나만 씀)
// 200t with limit(work as 14t): 3.3 - 0.5 = 2.8sec (코어 다 씀)
// 쓰레드가 hw thread max를 초과할 경우, 아예 병렬 처리 자체를 안 하는듯?