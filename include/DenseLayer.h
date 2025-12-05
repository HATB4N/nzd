#ifndef DENSELAYER_H
#define DENSELAYER_H

#include "Matrix.h"
#include "Activation.h"
#include "Struct.h"
#include "Initializers/IWeightInitializer.h"
#include <memory>
#include <vector>
#include <cstdint>

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
    Matrix_T<fp16>& get_weight();
    Matrix_T<fp16>& get_bias();
    void set_weight(const Matrix_T<fp16>& new_weights);
    void set_bias(const Matrix_T<fp16>& new_biases);
    uint8_t act_enum = 1; // 이거 임시임.
    // 초기화시에 활성화함수를 직접 받는게 아니라
    // 열거형으로 받아서 알아서 code영역 참조하게
    // 일단 했다고 가정.

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