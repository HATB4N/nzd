#ifndef HEINITIALIZER_H
#define HEINITIALIZER_H

#include "Initializers/IWeightInitializer.h"
#include <random>
#include <cmath>
#include <cassert>

class HeInitializer : public ParallelInitializer {
public:
    using ParallelInitializer::ParallelInitializer;

protected:
    void fill_chunk(std::span<fp16> chunk, uint64_t input_dim, uint64_t output_dim, uint32_t chunk_seed) const override {
        fp32 sigma = std::sqrt(2.0f / static_cast<fp32>(input_dim));
        
        std::mt19937 local_gen(chunk_seed);
        std::normal_distribution<fp32> local_dist(0.0f, sigma);

        for (auto& w : chunk) {
            w = static_cast<fp16>(local_dist(local_gen));
        }
    }
};

#endif // HEINITIALIZER_H

/*

class HeInitializer : public IWeightInitializer {
public:
    void initialize(Matrix_T<fp16>& weights, uint64_t input_dim, uint64_t output_dim) const override {
        assert(input_dim != 0);
        fp32 sigma = std::sqrt(2.0f / static_cast<fp32>(input_dim));
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<fp32> dist(0.0f, sigma);
        for(auto& w : weights.data(View::NT)) w = static_cast<fp16>(dist(gen));
    }
};

*/