#ifndef HEINITIALIZER_H
#define HEINITIALIZER_H

#include "IWeightInitializer.h"
#include <random>

// 테스트용
class HeInitializer : public IWeightInitializer {
public:
    void initialize(Matrix_T<fp16>& weights, size_t input_dim, size_t output_dim) const override {
        std::fill(weights.data(View::NT).begin(), weights.data(View::NT).end(), static_cast<fp16>(0.01f));
    }
};

#endif
