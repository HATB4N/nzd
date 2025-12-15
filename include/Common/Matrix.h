#ifndef MATRIX_H
#define MATRIX_H

#include "Common/Struct.h"
#include <cstddef>
#include "Common/Types.h"
#include <vector>
#include <span>
#include <future>
#include "Common/ThreadPool.h"
#include <cassert>

class Matrix {
public:
    Matrix(unsigned int threads = 14);

    void set_threads(unsigned int threads); // 나중에 work별 threads를 상위에서 유동적으로 조절하고 병렬로 실행 가능하게...?

    // r is (batch, out_dim), b is (1, out_dim)
    template<typename T_IN, typename T_OUT>
    void add_bias(Matrix_T<T_OUT> &r, const Matrix_T<T_IN> &b) {
        assert(r.col() == b.col());
        auto &r_data = r.data(View::NT);
        const auto &b_data = b.data(View::NT);

#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < r.row(); ++i) {
            for (size_t j = 0; j < r.col(); ++j) {
                r_data[i * r.col() + j] += static_cast<T_OUT>(b_data[j]);
            }
        }
    }

    template<typename T_IN, typename T_OUT>
    void add(Matrix_T<T_OUT> &r, const Matrix_T<T_IN> &b) {
        assert(r.row() == b.row() && r.col() == b.col());
        uint64_t sz = r.row() * r.col();

        auto &r_data = r.data(View::NT);
        const auto &b_data = b.data(View::NT);

#pragma omp parallel for simd schedule(static)
        for (uint64_t i = 0; i < sz; i++) {
            r_data[i] += static_cast<T_OUT>(b_data[i]);
        }
    }

    // r:= r - b // 일단 위는 가중치 sum용인데 얜 one hot용이니까 다르게 구현함.
    // r + 공통편향이었다면, 얜 그냥 component wise하게
    template<typename T_IN, typename T_OUT>
    void sub(Matrix_T<T_OUT> &r, const Matrix_T<T_IN> &b, fp32 lr = 1.0f) {
        assert(r.row() == b.row() && r.col() == b.col());
        uint64_t sz = r.size();

        auto &r_data = r.data(View::NT);
        const auto &b_data = b.data(View::NT);

#pragma omp parallel for simd schedule(static)
        for (uint64_t i = 0; i < sz; i++) {
            r_data[i] = static_cast<T_OUT>(static_cast<float>(r_data[i]) - lr * static_cast<float>(b_data[i]));
        }
    }

    // x is (batch, in_dim), w is (in_dim, out_dim), r is (batch, out_dim)
    template<typename TA, typename TB, typename TC>
    void multiply(Matrix_T<TC> &c, // res
                  const Matrix_T<TA> &a, // left hand
                  const Matrix_T<TB> &b, // right hand
                  View a_view = View::NT,
                  View b_view = View::NT) {
        const size_t M = a.row(a_view);
        const size_t K = a.col(a_view);
        const size_t K2 = b.row(b_view);
        const size_t N = b.col(b_view);
        assert(K == K2);
        assert(c.row() == M && c.col() == N);

        auto A = std::span<const TA>(a.data(a_view));
        auto BT = std::span<const TB>(b.data(b.flip(b_view)));
        auto C = std::span<TC>(c.data(View::NT));

        const size_t cs = M / _threads;
        size_t offset = 0;

        std::vector<std::future<void> > results;
        results.reserve(_threads);

        for (unsigned int i = 0; i < _threads; ++i) {
            const size_t curM = (i == _threads - 1) ? (M - offset) : cs;
            if (curM == 0) continue;

            auto A_part = A.subspan(offset * K, curM * K);
            auto C_part = C.subspan(offset * N, curM * N);

            results.emplace_back(
                ThreadPool::instance().enqueue([=, this]() {
                    this->mul_part<TA, TB, TC>(A_part, BT, C_part, curM, K, N);
                })
            );
            offset += curM;
        }

        for (auto &fut: results) {
            fut.get();
        }
    }

    template<typename T1, typename T2>
    void element_wise_multiply(Matrix_T<T1> &target, Matrix_T<T2> &other) {
        // std::cout << target.row() << ", " << target.col() << std::endl;
        // std::cout << other.row() << ", " << other.col() << std::endl;
        assert(target.row() == other.row() && target.col() == other.col());
        assert(target.size() == other.size());

        auto &target_data = target.data(View::NT);
        auto &other_data = other.data(View::NT);

        std::span<T1> target_span(target_data);
        std::span<const T2> other_span(other_data);

        uint64_t total_size = target.size();

        std::vector<std::future<void> > results;
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

        for (auto &fut: results) {
            fut.get();
        }
    }

private:
    // R = XW  <=> R = X(W^T)^T
    template<typename TA, typename TB, typename TC>
    void mul_part(std::span<const TA> A_part,
                  std::span<const TB> BT,
                  std::span<TC> C_part,
                  size_t curM, size_t K, size_t N) {
        for (size_t i = 0; i < curM; ++i) {
            const TA *a_row = &A_part[i * K];
            for (size_t j = 0; j < N; ++j) {
                const TB *bt_row = &BT[j * K];

                fp32 val = 0.0f;

#pragma omp simd reduction(+:val)
                for (size_t k = 0; k < K; ++k) {
                    val += static_cast<fp32>(a_row[k]) * static_cast<fp32>(bt_row[k]);
                }
                C_part[i * N + j] = static_cast<TC>(val);
            }
        }
    }

    template<typename T1, typename T2>
    void elem_mul_part(std::span<T1> target_chunk, std::span<const T2> other_chunk) {
        assert(target_chunk.size() == other_chunk.size());
        for (size_t i = 0; i < target_chunk.size(); ++i) {
            target_chunk[i] *= static_cast<T1>(other_chunk[i]);
        }
    }

    unsigned int _threads;
};

#endif // MATRIX_H
