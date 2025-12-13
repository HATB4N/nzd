#include "Optimizers/SGD.h"

SGD::SGD() {
    //
}

/*
void DenseLayer::update() {
    const float lr = 0.001f;
    
    auto& w_data = _weights.data(View::NT);
    const auto& gw_data = _grad_weights.data(View::NT);
    #pragma omp parallel for
    for (size_t i = 0; i < w_data.size(); ++i) {
        w_data[i] = static_cast<fp32>(static_cast<float>(w_data[i]) - lr * gw_data[i]);
    }

    auto& b_data = _biases.data(View::NT);
    const auto& gb_data = _grad_biases.data(View::NT);
    #pragma omp parallel for
    for (size_t i = 0; i < b_data.size(); ++i) {
        b_data[i] = static_cast<fp32>(static_cast<float>(b_data[i]) - lr * gb_data[i]);
    }

*/