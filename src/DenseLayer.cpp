#include "DenseLayer.h"
#include "Struct.h"
#include <algorithm>

DenseLayer::DenseLayer(
    unsigned int threads, 
    void(*act)(Matrix_T<fp32>&), 
    size_t input_dim, 
    size_t output_dim, 
    int idx, 
    std::unique_ptr<IWeightInitializer> initializer) 
    : _weights(input_dim, output_dim), _biases(1, output_dim), 
    _input_dim(input_dim), _output_dim(output_dim), 
    _initializer(std::move(initializer)) {
    _t = threads;
    _idx = idx;
    _act = act;
    _gemm = std::make_unique<Matrix>(_t);

    // init, 학습(중) 여부 따라 호출 여부 정하게.
    // 기본적으로 init시점에서(=load가 없는) parms init
    _initializer->initialize(_weights, _input_dim, _output_dim);
    _initializer->initialize(_biases, 1, _output_dim);
}

// R = σ(XW+b)
void DenseLayer::forward(const Matrix_T<fp16> &x, Matrix_T<fp32> &r) {
    _gemm->multiply(x, _weights, r);
    _gemm->add(r, _biases);
    _act(r);
}

// idk :V
// SGD쪽 구현 후에 진행
void DenseLayer::backward() {

}

// idk :V
// SGD쪽 구현 후에 진행
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