#include "Core/DenseLayer.h"
#include "Utils/Activation.h"
#include <algorithm>
#include <iostream>
#include "Common/Gemm.h"

DenseLayer::DenseLayer(ActType act_type,
                       uint64_t input_dim, 
                       uint64_t output_dim, 
                       InitType init,
                       OptType opt,
                       uint64_t idx) : _idx(idx),
                                       _input_dim(input_dim), _output_dim(output_dim),
                                       _weights(input_dim, output_dim), _biases(1, output_dim), 
                                       _grad_weights(input_dim, output_dim), _grad_biases(1, output_dim),
                                       _x_cache(0, 0), _z_cache(0, 0), // req reassign @ FW
                                       _act(resolve_act(act_type)), _act_difr(resolve_act_difr(act_type)),
                                       _initializer(std::move(resolve_init(init))), 
                                       _optimizer(resolve_opt(opt, _weights, _biases, _grad_weights, _grad_biases)),
                                       _act_type(act_type) {                          
    if (_initializer) { // allow nullptr
        _initializer->initialize(_weights, _input_dim, _output_dim);
        std::fill(_biases.data(View::NT).begin(), _biases.data(View::NT).end(), static_cast<fp32>(0.0f));
    }
    // if(!_optimizer) throw ... (nullptr exxception? idk)
    _runner = _bw_table[_act_type == ActType::SOFTMAX];
}

void DenseLayer::forward(const Matrix_T<fp32> &x, Matrix_T<fp32> &z) {
    _x_cache = x;
    gemm().multiply(z, x, _weights);
    gemm().add_bias<fp32, fp32>(z, _biases);
    _z_cache = z; // 용량 고려하면 backward시 x->r 계산 fallback도 고려
    _act(z);
}

void DenseLayer::backward(Matrix_T<fp32>& dR, Matrix_T<fp32>& dX) {
    (this->*_runner)(dR, dX);
}

void DenseLayer::update() {
    _optimizer->step();
}

void DenseLayer::_bw_impl_standard(Matrix_T<fp32>& dR, Matrix_T<fp32>& dX) {
    _act_difr(this->_z_cache); 
    gemm().element_wise_multiply<fp32, fp32>(dR, this->_z_cache);
    _compute_gradients(dR, dX); // dR := dZ
}

void DenseLayer::_bw_impl_bypass(Matrix_T<fp32>& dZ, Matrix_T<fp32>& dX) {
    _compute_gradients(dZ, dX);
}

void DenseLayer::_compute_gradients(Matrix_T<fp32>& dZ, Matrix_T<fp32>& dX) {
    gemm().multiply(_grad_weights, _x_cache, dZ, View::T, View::NT);
    gemm().multiply(dX, dZ, _weights, View::NT, View::T);
    _accum_bias_grad(dZ);
}

void DenseLayer::_accum_bias_grad(const Matrix_T<fp32>& dZ) {
    auto& gb = _grad_biases.data(View::NT);
    const auto& dz = dZ.data(View::NT);

    const size_t B = dZ.row();
    const size_t O = dZ.col();

    std::fill(gb.begin(), gb.end(), 0.0f);

    for (size_t j = 0; j < O; ++j) {
        fp32 s = 0;
        #pragma omp simd reduction(+:s)
        for (size_t i = 0; i < B; ++i) {
            s += dz[i * O + j];
        }
        gb[j] = s;
    }
}