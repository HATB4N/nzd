#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "Common/Struct.h"
#include "Common/Types.h"
#include <vector>
#include <cstddef>

enum class ActType : uint8_t {
    IDENTITY,
    SOFTMAX,
    SIGMOID,
    SILU,
    RELU,
    L_RELU
};

namespace Act {
    void inline identity(Matrix_T<fp32> &m1) { return; } // fix
    void softmax(Matrix_T<fp32> &m1);
    void sigmoid(Matrix_T<fp32> &m1);
    void silu(Matrix_T<fp32> &m1);
    void relu(Matrix_T<fp32> &m1);
    void l_relu(Matrix_T<fp32> &m1);
};

namespace ActDifr {
    void difr_identity(Matrix_T<fp32> &m1);
    void difr_softmax(Matrix_T<fp32> &m1);
    void difr_sigmoid(Matrix_T<fp32> &m1);
    void difr_silu(Matrix_T<fp32> &m1);
    void difr_relu(Matrix_T<fp32> &m1);
    void difr_l_relu(Matrix_T<fp32> &m1);
};

using ActFunc = void(*)(Matrix_T<fp32>&);

ActFunc resolve_act(ActType);
ActFunc resolve_act_difr(ActType);

#endif // ACTIVATION_H