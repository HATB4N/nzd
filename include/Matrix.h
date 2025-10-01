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
    Matrix(size_t m1_row, size_t m1_col, size_t m2_col, unsigned int threads);
    ~Matrix();
    void add(std::vector<fp32> &m_t, const std::vector<fp16> &b);
    void multiply(const std::vector<fp16> &w, const std::vector<fp16> &xt, 
        std::vector<fp32> &rt);
private:
    void product(std::span<const fp16> w, std::span<const fp16> xt, 
        std::span<fp32> rt, size_t batch_p);
    size_t _m1_row, _m1_col, _batch;
    unsigned int _threads;

};

#endif