#include "DenseLayer.h"
#include "Activation.h"
#include <algorithm>
#include <iostream>

DenseLayer::DenseLayer(ActFunc act_enum,
                       uint64_t input_dim, 
                       uint64_t output_dim, 
                       std::shared_ptr<IWeightInitializer> initializer,
                       uint64_t idx) : _idx(idx),
                                       _weights(input_dim, output_dim), _biases(1, output_dim), 
                                       _grad_weights(input_dim, output_dim), _grad_biases(1, output_dim), 
                                       _initializer(std::move(initializer)),
                                       _input_dim(input_dim), _output_dim(output_dim),
                                       _act(resolve_act(act_enum)), _act_difr(resolve_act_difr(act_enum)), 
                                       _gemm(std::make_unique<Matrix>()) {                          
    if (_initializer) { // allow nullptr
        _initializer->initialize(_weights, _input_dim, _output_dim);
        std::fill(_biases.data(View::NT).begin(), _biases.data(View::NT).end(), static_cast<fp16>(0.0f));
    }
}

// R = σ(XW+b), multiply(Y, X, W, View::T)
void DenseLayer::forward(const Matrix_T<fp16> &x, Matrix_T<fp32> &r) {
    // _x_cache = x;
    _gemm->multiply<fp16, fp32>(r, x, _weights);
    _gemm->add<fp16, fp32>(r, _biases);
    // _z_cache = r;
    _act(r);
}

// multiply(dX, dY, W, View::NT)
void DenseLayer::backward(Matrix_T<fp32>& d_out, Matrix_T<fp32>& d_in) {

}

// W := W - lr
void DenseLayer::update() {

}

Matrix_T<fp16>& DenseLayer::get_weight() {
    return this->_weights;
}

Matrix_T<fp16>& DenseLayer::get_bias() {
    return this->_biases;
}

Matrix_T<fp32>& DenseLayer::get_grad_weight() {
    return this->_grad_weights;
}

Matrix_T<fp32>& DenseLayer::get_grad_bias() {
    return this->_grad_biases;
}

void DenseLayer::set_weight(const Matrix_T<fp16>& new_weights) {
    this->_weights = new_weights;
}

void DenseLayer::set_bias(const Matrix_T<fp16>& new_biases) {
    this->_biases = new_biases;
}

ActFunc DenseLayer::get_act_func() {
    return this->act_func;
}
