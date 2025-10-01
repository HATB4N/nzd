#include "Matrix.h"
#include <cmath>
#include <algorithm>
#include <cassert>
#include <future>
#include <iostream>

void Matrix::setup(
    size_t m1_row, size_t m1_col, 
    size_t m2_row, size_t m2_col) {
    assert(m1_col == m2_row);
    _m1_row = m1_row;
    _m1_col = m1_col;
    _batch = m2_col;
    /*
    w: m1_row * m1_col
    x: m2_row * batch = m1_col * batch (논리적으로 전치)
    b: idk I should fix btw
    res: m1_row * batch (논리적으로 전치)
    */
    unsigned int MAX_T = std::thread::hardware_concurrency();
    _threads = std::min(_threads, MAX_T);
    if(!_threads) _threads = 1;
}

void Matrix::add(std::vector<fp32> &m_t, const std::vector<fp16> &b) {
    for(size_t i = 0; i< _batch; i++) {
        for(size_t j = 0; j< _m1_row; j++) {
            m_t[i*_m1_row + j] += static_cast<fp32>(b[j]);
        }
    }
}

void Matrix::multiply(
    const std::vector<fp16> &w, const std::vector<fp16> &xt, 
    std::vector<fp32> &rt) {

        auto _w = std::span<const fp16>(w);
        auto _xt = std::span<const fp16>(xt);
        auto _rt = std::span<fp32>(rt);

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

            auto _xt_part = std::span<const fp16>(_xt.data() + offset * _m1_col, bs * _m1_col);
            auto _rt_part = std::span<fp32>(_rt.data() + offset * _m1_row, bs * _m1_row);
            // async cowokers
            futures.emplace_back(std::async(std::launch::async, [=, this](){
                this->product(_w, _xt_part, _rt_part, bs);
            }));

            // update offset
            offset += bs;
        }

        // await (WHY E CORES)
        for (auto& fut : futures) {
            fut.get();
        }
    }

// WX => sum_by_row(W*(X^T))
void Matrix::product(
    std::span<const fp16> w, std::span<const fp16> xt_part, 
    std::span<fp32> rt_part, size_t batch_p) {

    for (size_t k = 0; k< batch_p; ++k) {
        const fp16* xk = &xt_part[k*_m1_col]; // 논리적으로 전치되어 저장된 x => 연속된 col에 접근
        for (size_t i = 0; i< _m1_row; ++i) {
            fp32 val = fp32(0.0f);
            const fp16* w_row = &w[i*_m1_col]; // W의 i행(연속)
            for (size_t j = 0; j< _m1_col; ++j) {
                val += static_cast<fp32>(w_row[j]) * static_cast<fp32>(xk[j]);
            }
            rt_part[k*_m1_row + i] = val; // 연속 블록 내 기록
        }
    }
}