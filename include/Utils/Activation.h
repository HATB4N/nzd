#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "Common/Struct.h"
#include "Common/Types.h"
#include <vector>
#include <cstddef>

enum class ActFunc : uint8_t {
    LINEAR,
    SOFTMAX,
    SIGMOID,
    SILU,
    RELU,
    L_RELU
};

namespace Act {
    void inline linear(Matrix_T<fp32> &m1) { return; } // fix
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

using ActFn = void(*)(Matrix_T<fp32>&);

ActFn resolve_act(ActFunc);
ActFn resolve_act_difr(ActFunc);

#endif // ACTIVATION_H