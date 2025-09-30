#ifndef SSAMMATH_H
#define SSAMMATH_H

#include <cstddef>
#include <stdfloat>
#include <vector>

// cpp23
using fp16 = std::float16_t;
using fp32 = std::float32_t;

/*
행렬곱
경사하강법
*/

class DenseLayer {
public:
    void setup(size_t m1_row, size_t m1_col, size_t m2_row);
    void forward(const std::vector<fp16> &w, const std::vector<fp16> &x, const std::vector<fp16> &b, std::vector<fp32> &res);

private:
    void softmax(std::vector<fp32> &m1);
    size_t row, col;

};

#endif