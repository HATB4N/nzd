#include "Activation.h"
#include <cassert>
#include <algorithm>
#include <cmath>

void Activation::setup(size_t m1_row, size_t m1_col, size_t m2_row, size_t m2_col) {
    // suppose that m2_col = 1
    // ignore batch
    assert(m1_col == m2_row);
    _m1_row = m1_row;
    _m1_col = m1_col;
    _batch = m2_col;
    /*
    m1_row * m1_col
    m1_col * 1 = m2_row * 1
    */
}

void Activation::softmax(std::vector<fp32> &m1_t) {
    // O(n)
    for(size_t i = 0; i< _m1_row; ++i) {
        auto start = m1_t.begin() + i * _batch;
        auto end = start + _batch;
        fp32 max = *std::max_element(start, end);
        fp32 sum = 0.0f;

        for(auto cur_m = start; cur_m< end; cur_m++) {
            *cur_m = std::exp(*cur_m - max);
            sum += *cur_m;
        }

        fp32 inv = 1.0f / sum;
        for(auto cur_m = start; cur_m< end; cur_m++) {
            *cur_m *= inv;
        }
    }
}

void Activation::sigmoid(std::vector<fp32> &m1_t) {

}

void Activation::silu(std::vector<fp32> &m1_t) {
    
}

