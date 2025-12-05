#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "Struct.h"
#include <stdfloat>
#include <vector>
#include <cstddef>

using fp16 = std::float16_t;
using fp32 = std::float32_t;

enum Func { // 이거 받아서 DL에서 알아서 함수포인터 찾든 해서 초기화하게.
    // 컨스트럭터 수정해야 함.
    LINEAR,
    SOFTMAX,
    SIGMOID,
    SILU,
    RELU,
    L_RELU
};

namespace Act {
    void inline linear(Matrix_T<fp32> &m1) { return; }
    void softmax(Matrix_T<fp32> &m1);
    void sigmoid(Matrix_T<fp32> &m1);
    void silu(Matrix_T<fp32> &m1);
    void relu(Matrix_T<fp32> &m1);
    void l_relu(Matrix_T<fp32> &m1);
};

namespace ActDifr {
    void difr_linear(Matrix_T<fp32> &m1);
    void difr_softmax(Matrix_T<fp32> &m1);
    void difr_sigmoid(Matrix_T<fp32> &m1);
    void difr_silu(Matrix_T<fp32> &m1);
    void difr_relu(Matrix_T<fp32> &m1);
    void difr_l_relu(Matrix_T<fp32> &m1);
};

#endif