#ifndef MATRIX_H
#define MATRIX_H

#include "Struct.h"
#include <cstddef>
#include <stdfloat>
#include <vector>
#include <span>

// cpp23
using fp16 = std::float16_t;
using fp32 = std::float32_t;

class Matrix { // 생성자, 소멸자로 thread pool을 static하게 두는 것도 좋을 듯?
public:
    Matrix(unsigned int threads);
    ~Matrix();
    void add(Matrix_T<fp32> &m_t, const Matrix_T<fp16> &b);
    void multiply(const Matrix_T<fp16> &w, const Matrix_T<fp16> &x, 
        Matrix_T<fp32> &r);
private:
    void mul_part(std::span<const fp16> w, std::span<const fp16> xt, 
        std::span<fp32> rt, size_t batch_p, size_t _w_row, size_t _w_col);
    unsigned int _threads;

};

#endif