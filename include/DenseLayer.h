#ifndef DENSELAYER_H
#define DENSELAYER_H

#include "Matrix.h"
#include "Activation.h"
#include <memory>
#include <vector>

using fp16 = std::float16_t;
using fp32 = std::float32_t;

class DenseLayer {
public:
    DenseLayer(unsigned int threads, void(*act)(Matrix_T<fp32>&));
    ~DenseLayer() = default;
    void forward(Matrix_T<fp16> &w, Matrix_T<fp16> &x, Matrix_T<fp16> &b, Matrix_T<fp32> &r);
    void backward();
    void update();

private:
    std::unique_ptr<Matrix> _gemm;
    void (*_act)(Matrix_T<fp32> &);
    unsigned int _t;
};

#endif