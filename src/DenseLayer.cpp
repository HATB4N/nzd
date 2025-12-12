#include "DenseLayer.h"
#include "Activation.h"
#include <algorithm>
#include <iostream>
#include "Common/Gemm.h"

DenseLayer::DenseLayer(ActFunc act_enum,
                       uint64_t input_dim, 
                       uint64_t output_dim, 
                       std::shared_ptr<IWeightInitializer> initializer,
                       uint64_t idx) : _idx(idx),
                                       _input_dim(input_dim), _output_dim(output_dim),
                                       _weights(input_dim, output_dim), _biases(1, output_dim), 
                                       _grad_weights(input_dim, output_dim), _grad_biases(1, output_dim),
                                       _x_cache(input_dim, output_dim), _z_cache(input_dim, output_dim), 
                                       _act(resolve_act(act_enum)), _act_difr(resolve_act_difr(act_enum)),
                                       _initializer(std::move(initializer)), _act_func(act_enum) {                          
    if (_initializer) { // allow nullptr
        _initializer->initialize(_weights, _input_dim, _output_dim);
        std::fill(_biases.data(View::NT).begin(), _biases.data(View::NT).end(), static_cast<fp16>(0.0f));
    }
    if(_act_func == ActFunc::SOFTMAX) { // unlikely
        _backward_runner = &DenseLayer::_bw_impl_bypass;
    } else { // likely
        _backward_runner = &DenseLayer::_bw_impl_standard;
    }
}
// R = σ(XW+b), multiply(Y, X, W, View::T)
void DenseLayer::forward(const Matrix_T<fp16> &x, Matrix_T<fp32> &r) {
    _x_cache = x;
    gemm().multiply<fp16, fp32>(r, x, _weights);
    gemm().add_bias<fp16, fp32>(r, _biases);
    _z_cache = r; // 용량 고려하면 gemm().multiply<fp16, fp32>(_z_cache, _x_cache, _weights);이 나을지도...
    _act(r);
}

// multiply(dX, dY, W, View::NT)
void DenseLayer::backward(Matrix_T<fp32>& dR, Matrix_T<fp32>& dX) {
    (this->*_backward_runner)(dR, dX);
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
    return this->_act_func;
}

void DenseLayer::_bw_impl_standard(Matrix_T<fp32>& dR, Matrix_T<fp32>& dX) {
    _act_difr(this->_z_cache); 
    gemm().element_wise_multiply<fp32, fp32>(dR, this->_z_cache);
    _compute_gradients(dR, dX); 
}

void DenseLayer::_bw_impl_bypass(Matrix_T<fp32>& dR, Matrix_T<fp32>& dX) {
    _compute_gradients(dR, dX);
}

void DenseLayer::_compute_gradients(Matrix_T<fp32>& dR, Matrix_T<fp32>& dX) {
    return;
}