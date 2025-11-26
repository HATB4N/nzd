#ifndef DENSELAYER_H
#define DENSELAYER_H

#include "Matrix.h"
#include "Activation.h"
#include "Struct.h"
#include "Initializers/IWeightInitializer.h"
#include <memory>
#include <vector>

using fp16 = std::float16_t;
using fp32 = std::float32_t;

class DenseLayer {
public:
    DenseLayer(unsigned int threads, 
        void(*act)(Matrix_T<fp32>&), 
        size_t input_dim, 
        size_t output_dim, 
        int idx,
        std::unique_ptr<IWeightInitializer> initializer);
    ~DenseLayer() = default;
    void forward(const Matrix_T<fp16> &x, Matrix_T<fp32> &r);
    void backward();
    void update();

private:
    Matrix_T<fp16> _weights;
    Matrix_T<fp16> _biases;
    size_t _input_dim;
    size_t _output_dim;

    std::unique_ptr<Matrix> _gemm;
    void (*_act)(Matrix_T<fp32> &);
    unsigned int _t;
    int _idx;
    std::unique_ptr<IWeightInitializer> _initializer;
};

#endif