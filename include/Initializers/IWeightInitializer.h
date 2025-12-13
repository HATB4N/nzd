#ifndef IWEIGHTINITIALIZER_H
#define IWEIGHTINITIALIZER_H

#include "Common/Struct.h"
#include "Common/Types.h"
#include "Common/ThreadPool.h"
#include <thread>
#include <span>

class IWeightInitializer {
public:
    virtual void initialize(Matrix_T<fp32>& weights, uint64_t input_dim, uint64_t output_dim) const = 0;
    virtual ~IWeightInitializer() = default;
};

class ParallelInitializer : public IWeightInitializer {
public:
    explicit ParallelInitializer(uint32_t seed, unsigned int threads = 14) : _seed(seed) {
            unsigned int MAX_T = std::thread::hardware_concurrency();
            if(!threads) threads = 1;
            this->_threads = std::min(threads, MAX_T);
        }

    void initialize(Matrix_T<fp32>& weights, uint64_t input_dim, uint64_t output_dim) const override final {
        auto &raw_data = weights.data(View::NT);
        std::span<fp32> data_span(raw_data);
        uint64_t total_size = data_span.size();

        std::vector<std::future<void>> results;
        uint64_t chunk_size = total_size / _threads;
        uint64_t offset = 0;

        for (unsigned int i = 0; i < _threads; ++i) {
            uint64_t current_size = (i == _threads - 1) ? (total_size - offset) : chunk_size;
            if (current_size == 0) continue;

            auto sub_span = data_span.subspan(offset, current_size);
            uint32_t chunk_seed = _seed + i;

            results.emplace_back(
                ThreadPool::instance().enqueue([=, this]() {
                    this->fill_chunk(sub_span, input_dim, output_dim, chunk_seed);
                })
            );
            offset += current_size;
        }
        for (auto& fut : results) fut.get();
    }
protected:
    unsigned int _threads;
    uint32_t _seed;
    virtual void fill_chunk(std::span<fp32> chunk, uint64_t input_dim, uint64_t output_dim, uint32_t chunk_seed) const = 0;
};

#endif // IWEIGHTINITIALIZER_H