#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "Struct.h"
#include <stdfloat>
#include <vector>
#include <cstddef>

using fp16 = std::float16_t;
using fp32 = std::float32_t;

namespace Act {
    void softmax(Matrix_T<fp32> &m1);
    void sigmoid(Matrix_T<fp32> &m1);
    void silu(Matrix_T<fp32> &m1);
    void relu(Matrix_T<fp32> &m1);
    void l_relu(Matrix_T<fp32> &m1);
};

namespace ActDifr {
    void difr_softmax(Matrix_T<fp32> &m1);
    void difr_sigmoid(Matrix_T<fp32> &m1);
    void difr_silu(Matrix_T<fp32> &m1);
    void difr_relu(Matrix_T<fp32> &m1);
    void difr_l_relu(Matrix_T<fp32> &m1);
};

#endif