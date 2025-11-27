#ifndef HEINITIALIZER_H
#define HEINITIALIZER_H

#include "IWeightInitializer.h"
#include <random>
#include <cmath>
#include <cassert>

class HeInitializer : public IWeightInitializer {
public:
    void initialize(Matrix_T<fp16>& weights, size_t input_dim, size_t output_dim) const override {
        assert(input_dim != 0);
        fp32 sigma = std::sqrt(2.0f / static_cast<fp32>(input_dim));
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<fp32> dist(0.0f, sigma);
        for(auto& w : weights.data(View::NT)) w = static_cast<fp16>(dist(gen));
    }
};

#endif
