#ifndef MATRIX_H
#define MATRIX_H

#include "Common/Struct.h"
#include <cstddef>
#include "Common/Types.h"
#include <vector>
#include <span>
#include <future>
#include "ThreadPool.h"
#include <cassert>

class Matrix {
public:
    Matrix(unsigned int threads = 14);
    ~Matrix();
    void set_threads(unsigned int threads); // 나중에 work별 threads를 상위에서 유동적으로 조절하고 병렬로 실행 가능하게...?

    // r is (batch, out_dim), b is (1, out_dim)
    template <typename T_IN, typename T_OUT>
    void add(Matrix_T<T_OUT> &r, const Matrix_T<T_IN> &b) {
        auto& r_data = r.data(View::NT);
        const auto& b_data = b.data(View::NT);

        for(size_t i = 0; i < r.row(); ++i) {
            for(size_t j = 0; j < r.col(); ++j) {
                r_data[i * r.col() + j] += static_cast<T_OUT>(b_data[j]);
            }
        }
    }

    // x is (batch, in_dim), w is (in_dim, out_dim), r is (batch, out_dim)
    template <typename T_IN, typename T_OUT>
    void multiply(Matrix_T<T_OUT> &r, const Matrix_T<T_IN> &x, const Matrix_T<T_IN> &w, View x_view_type = View::NT,  View w_view_type = View::T) {
        auto _x = std::span<const T_IN>(x.data(x_view_type));
        auto _w = std::span<const T_IN>(w.data(w_view_type));
        auto _r = std::span<T_OUT>(r.data(x_view_type)); // 불확실

        size_t batch_size = (x_view_type == View::T) ? x.col() : x.row();
        size_t in_dim = (x_view_type == View::T) ? x.row() : x.col();
        size_t out_dim = (w_view_type == View::NT) ? w.col() : w.row();

        std::vector<std::future<void>> results;
        results.reserve(_threads);

        const size_t cs = batch_size / _threads;
        size_t offset = 0;

        for (unsigned int i = 0; i < _threads; ++i) {
            const size_t current_batch_size = (i == _threads - 1) ? (batch_size - offset) : cs;
            if (current_batch_size == 0) continue;

            auto _x_part = _x.subspan(offset * in_dim, current_batch_size * in_dim);
            auto _r_part = _r.subspan(offset * out_dim, current_batch_size * out_dim);
            
            results.emplace_back(
                ThreadPool::instance().enqueue([=, this]() {
                    this->mul_part<T_IN, T_OUT>(_x_part, _w, _r_part, current_batch_size, in_dim, out_dim);
                })
            );
            offset += current_batch_size;
        }

        for (auto& fut : results) {
            fut.get();
        }
    }

    template <typename T1, typename T2>
    void element_wise_multiply(Matrix_T<T1>& target, Matrix_T<T2>& other) {
        assert(target.size() == other.size());

        auto& target_data = target.data(View::NT);
        auto& other_data = other.data(View::NT);

        std::span<T1> target_span(target_data);
        std::span<const T2> other_span(other_data);

        uint64_t total_size = target.size();

        std::vector<std::future<void>> results;
        results.reserve(_threads);

        uint64_t chunk_size = total_size / _threads;
        uint64_t offset = 0;

        for (unsigned int i = 0; i < _threads; ++i) {
            const uint64_t current_size = (i == _threads - 1) ? (total_size - offset) : chunk_size;
            if (current_size == 0) continue;

            auto target_sub_span = target_span.subspan(offset, current_size);
            auto other_sub_span = other_span.subspan(offset, current_size);
            
            results.emplace_back(
                ThreadPool::instance().enqueue([=, this]() {
                    this->elem_mul_part(target_sub_span, other_sub_span);
                })
            );
            offset += current_size;
        }

        for (auto& fut : results) {
            fut.get();
        }
    }

private:
    // R = XW  <=> R = X(W^T)^T
    template <typename T_IN, typename T_OUT>
    void mul_part(std::span<const T_IN> x_part, 
                  std::span<const T_IN> w_transposed, 
                  std::span<T_OUT> r_part, 
                  size_t x_rows, 
                  size_t in_dim, 
                  size_t out_dim) {
        for (size_t i = 0; i < x_rows; ++i) {
            const T_IN* x_row = &x_part[i * in_dim];
            for (size_t j = 0; j < out_dim; ++j) {
                const T_IN* wt_row = &w_transposed[j * in_dim];
                
                T_OUT val = static_cast<T_OUT>(0); 

                #pragma omp simd reduction(+:val)
                for (size_t k = 0; k < in_dim; ++k) {
                    val += static_cast<T_OUT>(x_row[k]) * static_cast<T_OUT>(wt_row[k]);
                }
                r_part[i * out_dim + j] = val;
            }
        }
    }

    template <typename T1, typename T2>
    void elem_mul_part(std::span<T1> target_chunk, std::span<const T2> other_chunk) {
        assert(target_chunk.size() == other_chunk.size());
        for (size_t i = 0; i < target_chunk.size(); ++i) {
            target_chunk[i] *= static_cast<T1>(other_chunk[i]);
        }
    }

    unsigned int _threads;

};

#endif // MATRIX_H