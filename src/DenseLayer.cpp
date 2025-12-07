#include "DenseLayer.h"
#include <algorithm>

DenseLayer::DenseLayer(ActFunc act_enum,
                       uint64_t input_dim, 
                       uint64_t output_dim, 
                       std::shared_ptr<IWeightInitializer> initializer,
                       uint64_t idx) : _idx(idx), // unique identifier of each layers. 나중에 구조체에 id - layer묶어서 관리하는게 편할듯. 일단 임시 ㅇㅇ
                                       _weights(input_dim, output_dim), _biases(1, output_dim), 
                                       _grad_weights(input_dim, output_dim), _grad_biases(1, output_dim), 
                                       _initializer(std::move(initializer)),
                                       _input_dim(input_dim), _output_dim(output_dim),
                                       _act(resolve_act(act_enum)), _act_difr(resolve_act_difr(act_enum)), 
                                       _gemm(std::make_unique<Matrix>()) {                          
    if (_initializer) {
        _initializer->initialize(_weights, _input_dim, _output_dim);
        // _initializer->initialize(_biases, 1, _output_dim);
        std::fill(_biases.data(View::NT).begin(), _biases.data(View::NT).end(), static_cast<fp16>(0.0f));
    }
}

// R = σ(XW+b)
void DenseLayer::forward(const Matrix_T<fp16> &x, Matrix_T<fp32> &r) {
    _gemm->multiply<fp16, fp32>(r, x, _weights, View::T); // for r(0..n) r_i := <x, w_i>
    // multiply(Y, X, W, View::T)
    _gemm->add<fp16, fp32>(r, _biases); // r := r + b
    _act(r); // r := σ(r)
}

Matrix_T<fp32> DenseLayer::backward(const Matrix_T<fp32> &grad_output) {
    // _gemm->multiply<fp32, fp32>(..., View::NT)
    // multiply(dX, dY, W, View::NT)
}

void DenseLayer::update() {

}

Matrix_T<fp16>& DenseLayer::get_weight() {
    return this->_weights;
}

Matrix_T<fp16>& DenseLayer::get_bias() {
    return this->_biases;
}

void DenseLayer::set_weight(const Matrix_T<fp16>& new_weights) {
    _weights = new_weights;
}

void DenseLayer::set_bias(const Matrix_T<fp16>& new_biases) {
    _biases = new_biases;
}

ActFunc DenseLayer::get_act_func() {
    return this->act_func;
}