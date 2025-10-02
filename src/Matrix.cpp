#include "Matrix.h"
#include <cmath>
#include <algorithm>
#include <future>

Matrix::Matrix(unsigned int threads) {
    unsigned int MAX_T = std::thread::hardware_concurrency();
    _threads = std::min(threads, MAX_T);
    if(!_threads) _threads = 1;
}

Matrix::~Matrix() {
    // 쓰레드 풀 정리 관련
}

void Matrix::add(Matrix_T<fp32> &m_t, const Matrix_T<fp16> &b) {
    for(size_t i = 0; i< m_t.col(); ++i) {
        for(size_t j = 0; j< m_t.row(); ++j) {
            m_t.data(View::T)[i*m_t.row() + j] += static_cast<fp32>(b.data(View::NT)[j]);
        }
    }
}

void Matrix::multiply(const Matrix_T<fp16> &w, const Matrix_T<fp16> &x, Matrix_T<fp32> &r) {
     // unique_ptr로 살아있는 동안 쓰레드 풀 자체는 유지되게. 메모리 관리 위해 ~Matrix 정의 해둘 것.
    auto _w = std::span<const fp16>(w.data(View::NT));
    auto _xt = std::span<const fp16>(x.data(View::T));
    auto _rt = std::span<fp32>(r.data(View::T));
    size_t _batch = x.col();
    size_t _w_row = w.row();
    size_t _w_col = w.col();

    std::vector<std::future<void>> futures;
    futures.reserve(_threads);

    const size_t cs = _batch / _threads;
    size_t offset = 0;
    for (unsigned int i = 0; i< _threads; ++i) {
        // bs = current batch size
        // offset = current offset
        // cs = base chunk size
        const size_t bs = (i == _threads - 1) ? (_batch - offset) : cs;
        if (bs == 0) continue; // 처리할 작업이 없으면 건너뜀

        auto _xt_part = std::span<const fp16>(_xt.data() + offset * _w_col, bs * _w_col);
        auto _rt_part = std::span<fp32>(_rt.data() + offset * _w_row, bs * _w_row);
        // async cowokers
        futures.emplace_back(std::async(std::launch::async, [=, this](){
            this->mul_part(_w, _xt_part, _rt_part, bs, _w_row, _w_col);
        }));

        // update offset
        offset += bs;
    }
    // await (WHY E CORES)
    for (auto& fut : futures) {
        fut.get();
    }
}

// R^T = W*(X^T)
void Matrix::mul_part(
    std::span<const fp16> w, std::span<const fp16> xt_part, 
    std::span<fp32> rt_part, size_t batch_p,
    size_t _w_row, size_t _w_col) {
    for (size_t k = 0; k< batch_p; ++k) {
        const fp16* xk = &xt_part[k*_w_col]; // 논리적으로 전치되어 저장된 x => 연속된 col에 접근
        for (size_t i = 0; i< _w_row; ++i) {
            fp32 val = fp32(0.0f);
            const fp16* wr = &w[i*_w_col]; // W의 i행(연속)
            for (size_t j = 0; j< _w_col; ++j) {
                val += static_cast<fp32>(wr[j]) * static_cast<fp32>(xk[j]);
            }
            rt_part[k*_w_row + i] = val; // 연속 블록 내 기록
        }
    }
}