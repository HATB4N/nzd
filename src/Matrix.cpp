#include "Matrix.h"
#include <cmath>
#include <algorithm>
#include <future>
#include "ThreadPool.h"

Matrix::Matrix(unsigned int threads) {
    unsigned int MAX_T = std::thread::hardware_concurrency();
    _threads = std::min(threads, MAX_T);
    if(!_threads) _threads = 1;
}

Matrix::~Matrix() {
    // 나중에 쓰레드풀 정적으로 두고 여기서 정리시키기
}

// r is (batch, out_dim), b is (1, out_dim)
void Matrix::add(Matrix_T<fp32> &r, const Matrix_T<fp16> &b) {
    auto r_data = r.data(View::NT);
    auto b_data = b.data(View::NT);

    for(size_t i = 0; i < r.row(); ++i) {
        for(size_t j = 0; j < r.col(); ++j) {
            r_data[i * r.col() + j] += static_cast<fp32>(b_data[j]);
        }
    }
}

// x is (batch, in_dim), w is (in_dim, out_dim), r is (batch, out_dim)
void Matrix::multiply(const Matrix_T<fp16> &x, const Matrix_T<fp16> &w, Matrix_T<fp32> &r) {
    auto _x = std::span<const fp16>(x.data(View::NT));
    auto _wt = std::span<const fp16>(w.data(View::T));
    auto _r = std::span<fp32>(r.data(View::NT));

    size_t batch_size = x.row();
    size_t in_dim = x.col();
    size_t out_dim = w.col();

    // wait targets
    std::vector<std::future<void>> results;
    results.reserve(_threads); // split job

    const size_t cs = batch_size / _threads;
    size_t offset = 0;

    for (unsigned int i = 0; i < _threads; ++i) {
        const size_t current_batch_size = (i == _threads - 1) ? (batch_size - offset) : cs;
        if (current_batch_size == 0) continue;

        auto _x_part = std::span<const fp16>(_x.data() + offset * in_dim, current_batch_size * in_dim);
        auto _r_part = std::span<fp32>(_r.data() + offset * out_dim, current_batch_size * out_dim);
        
        results.emplace_back(
            ThreadPool::instance().enqueue([=, this]() {
                this->mul_part(_x_part, _wt, _r_part, current_batch_size, in_dim, out_dim);
            })
        );
        offset += current_batch_size;
    }

    // synchonization
    for (auto& fut : results) {
        fut.get();
    }
}

// R = XW  <=> R = <X, (W^T)>
void Matrix::mul_part(
    std::span<const fp16> x_part, std::span<const fp16> wt, 
    std::span<fp32> r_part, size_t x_rows, size_t in_dim, size_t out_dim) {

    for (size_t i = 0; i < x_rows; ++i) { // X row중에 할당받는 부분
        const fp16* x_row = &x_part[i * in_dim];
        for (size_t j = 0; j < out_dim; ++j) { // W col = W^T row
            const fp16* wt_row = &wt[j * in_dim];
            fp32 val = fp32(0.0f);
            for (size_t k = 0; k < in_dim; ++k) { // 내?적
                val += static_cast<fp32>(x_row[k]) * static_cast<fp32>(wt_row[k]);
            }
            r_part[i * out_dim + j] = val;
        }
    }
}