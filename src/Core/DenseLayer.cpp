#include "Core/DenseLayer.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>

#include "Common/Gemm.h"
#include "Utils/Activation.h"

DenseLayer::DenseLayer(uint64_t input_dim, uint64_t output_dim,
                       ActType act_type, InitType init, OptType opt,
                       uint64_t idx) : idx_(idx),
                                       input_dim_(input_dim),
                                       output_dim_(output_dim),
                                       weights_(input_dim, output_dim),
                                       biases_(1, output_dim),
                                       grad_weights_(input_dim, output_dim),
                                       grad_biases_(1, output_dim),
                                       x_cache_(0, 0),
                                       z_cache_(0, 0),
                                       act_(resolve_act(act_type)),
                                       act_difr_(resolve_act_difr(act_type)),
                                       initializer_(std::move(resolve_init(init))),
                                       optimizer_(
                                           resolve_opt(opt, weights_, biases_, grad_weights_, grad_biases_)),
                                       act_type_(act_type) {
    if (!optimizer_)
        throw std::invalid_argument(
            "Failed to init denselayer: Invalid optimizer type");
    if (initializer_) {
        // allow nullptr
        initializer_->initialize(weights_, input_dim_, output_dim_);
        std::fill(biases_.data(View::NT).begin(), biases_.data(View::NT).end(),
                  static_cast<fp32>(0.0f));
    }
    runner_ = bw_table_[act_type_ == ActType::SOFTMAX];
}

void DenseLayer::forward(const Matrix_T<fp32> &X, Matrix_T<fp32> &Z) {
    x_cache_ = X;
    gemm().multiply(Z, X, weights_);
    gemm().add_bias<fp32, fp32>(Z, biases_);
    z_cache_ = Z; // 용량 고려하면 backward시 x->r 계산 fallback도 고려
    act_(Z);
}

void DenseLayer::backward(Matrix_T<fp32> &dR, Matrix_T<fp32> &dX) {
    (this->*runner_)(dR, dX);
}

void DenseLayer::update() { optimizer_->step(); }

void DenseLayer::bw_impl_standard_(Matrix_T<fp32> &dR, Matrix_T<fp32> &dX) {
    act_difr_(this->z_cache_);
    gemm().element_wise_multiply<fp32, fp32>(dR, this->z_cache_);
    compute_gradients_(dR, dX); // dR := dZ
}

void DenseLayer::bw_impl_bypass_(Matrix_T<fp32> &dZ, Matrix_T<fp32> &dX) {
    compute_gradients_(dZ, dX);
}

void DenseLayer::compute_gradients_(Matrix_T<fp32> &dZ, Matrix_T<fp32> &dX) {
    gemm().multiply(grad_weights_, x_cache_, dZ, View::T, View::NT);
    gemm().multiply(dX, dZ, weights_, View::NT, View::T);
    accum_bias_grad_(dZ);
}

void DenseLayer::accum_bias_grad_(const Matrix_T<fp32> &dZ) {
    auto &gb = grad_biases_.data(View::NT);
    const auto &dz = dZ.data(View::NT);
    const size_t B = dZ.row();
    const size_t O = dZ.col();
    std::fill(gb.begin(), gb.end(), 0.0f);
    for (size_t j = 0; j < O; ++j) {
        fp32 s = 0;
#pragma omp simd reduction(+ : s)
        for (size_t i = 0; i < B; ++i) {
            s += dz[i * O + j];
        }
        gb[j] = s;
    }
}
