#ifndef MATRIX_H
#define MATRIX_H

#include "Common/Struct.h"
#include <cstddef>
#include "Common/Types.h"
#include <vector>
#include <span>
#include <future>
#include "ThreadPool.h"

class Matrix {
public:
    Matrix(unsigned int threads = 14);
    ~Matrix();
    void set_threads(unsigned int threads); // 나중에 work별 threads를 상위에서 유동적으로 조절하고 병렬로 실행 가능하게...?

    // r is (batch, out_dim), b is (1, out_dim)
    template <typename T_IN, typename T_OUT>
    void add(Matrix_T<T_OUT> &r, const Matrix_T<T_IN> &b) {
        auto r_data = r.data(View::NT);
        auto b_data = b.data(View::NT);

        for(size_t i = 0; i < r.row(); ++i) {
            for(size_t j = 0; j < r.col(); ++j) {
                r_data[i * r.col() + j] += static_cast<T_OUT>(b_data[j]);
            }
        }
    }

    // x is (batch, in_dim), w is (in_dim, out_dim), r is (batch, out_dim)
    template <typename T_IN, typename T_OUT>
    void multiply(Matrix_T<T_OUT> &r, const Matrix_T<T_IN> &x, const Matrix_T<T_IN> &w, View w_view_type = View::T) {
        auto _x = std::span<const T_IN>(x.data(View::NT));
        auto _wt = std::span<const T_IN>(w.data(w_view_type));; // for backpropa
        auto _r = std::span<T_OUT>(r.data(View::NT));

        size_t batch_size = x.row();
        size_t in_dim = x.col();
        size_t out_dim = w.col();

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
                    this->mul_part<T_IN, T_OUT>(_x_part, _wt, _r_part, current_batch_size, in_dim, out_dim);
                })
            );
            offset += current_batch_size;
        }

        for (auto& fut : results) {
            fut.get();
        }
    }
private:
    // R = XW  <=> R = X(W^T)^T
    template <typename T_IN, typename T_OUT>
    void mul_part(
        std::span<const T_IN> x_part, std::span<const T_IN> wt, 
        std::span<T_OUT> r_part, size_t x_rows, size_t in_dim, size_t out_dim) {

        for (size_t i = 0; i < x_rows; ++i) {
            const T_IN* x_row = &x_part[i * in_dim];
            for (size_t j = 0; j < out_dim; ++j) {
                const T_IN* wt_row = &wt[j * in_dim];
                
                T_OUT val = static_cast<T_OUT>(0); 

                #pragma omp simd reduction(+:val)
                for (size_t k = 0; k < in_dim; ++k) {
                    val += static_cast<T_OUT>(x_row[k]) * static_cast<T_OUT>(wt_row[k]);
                }
                r_part[i * out_dim + j] = val;
            }
        }
    }
    unsigned int _threads;

};

#endif