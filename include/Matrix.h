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
    void add(Matrix_T<fp32> &r, const Matrix_T<fp16> &b);
    void multiply(const Matrix_T<fp16> &x, const Matrix_T<fp16> &w, 
        Matrix_T<fp32> &r);
private:
    void mul_part(std::span<const fp16> x, std::span<const fp16> w, 
        std::span<fp32> r, size_t x_row, size_t in_dim, size_t out_dim);
    unsigned int _threads;

};

#endif