#ifndef DENSELAYER_H
#define DENSELAYER_H

#include "Matrix.h"
#include "Activation.h"
#include "Common/Struct.h"
#include "Common/Types.h"
#include "Initializers/IWeightInitializer.h"
#include <memory>
#include <vector>
#include <cstdint>

class DenseLayer {
public:
    DenseLayer(ActFunc act_enum,
               uint64_t input_dim, 
               uint64_t output_dim, 
               std::shared_ptr<IWeightInitializer> initializer,
               uint64_t idx); // 얘는 _layers에서의 index(physical id)임.
    ~DenseLayer() = default;

    void forward(const Matrix_T<fp16> &x, Matrix_T<fp32> &r);
    Matrix_T<fp32> backward(const Matrix_T<fp32> &grad_output);
    void update();

    Matrix_T<fp16>& get_weight();
    Matrix_T<fp16>& get_bias();
    Matrix_T<fp32>& get_grad_weight();
    Matrix_T<fp32>& get_grad_bias();
    void set_weight(const Matrix_T<fp16>& new_weights);
    void set_bias(const Matrix_T<fp16>& new_biases);
    ActFunc get_act_func();
    

private:
    Matrix_T<fp16> _weights;
    Matrix_T<fp16> _biases;
    Matrix_T<fp32> _grad_weights;
    Matrix_T<fp32> _grad_biases;
    // Matrix_T<fp16> _x_cache;
    uint64_t _input_dim;
    uint64_t _output_dim;
    
    ActFunc act_func;

    std::unique_ptr<Matrix> _gemm;
    void (*_act)(Matrix_T<fp32> &);
    void (*_act_difr)(Matrix_T<fp32> &);
    uint64_t _idx;
    std::shared_ptr<IWeightInitializer> _initializer;
};

#endif