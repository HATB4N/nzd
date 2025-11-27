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
    _initializer->initialize(_weights, _input_dim, _output_dim);
    
    // 일단 b 0으로
    std::fill(_biases.data(View::NT).begin(), _biases.data(View::NT).end(), static_cast<fp16>(0.0f));
}

// R = σ(XW+b)
void DenseLayer::forward(const Matrix_T<fp16> &x, Matrix_T<fp32> &r) {
    _gemm->multiply(x, _weights, r);
    _gemm->add(r, _biases);
    _act(r);
}

// idk :V
void DenseLayer::backward() {

}

// idk :V
void DenseLayer::update() {

}