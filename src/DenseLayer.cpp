#include "DenseLayer.h"
#include <cmath>
#include <algorithm>
#include <cassert>

void DenseLayer::setup(size_t m1_row, size_t m1_col, size_t m2_row) {
    // suppose that m2_col = 1
    // ignore batch
    assert(m1_col == m2_row);
    row = m1_row;
    col = m1_col;
    /*
    m1_row * m1_col
    m1_col * 1 = m2_row * 1
    */
}

void DenseLayer::forward(
    const std::vector<fp16> &w, const std::vector<fp16> &x, 
    const std::vector<fp16> &b, std::vector<fp32> &res) {
    // O(n^2)
    // 행렬곱 결과 res에 저장
    for(size_t i = 0; i< row; i++) {
        for(size_t j = 0; j< col; j++) {
            res[i] += w[i*col+j] * x[j];
        }
        res[i] += b[i];
    }
    softmax(res);
}

void DenseLayer::softmax(std::vector<fp32> &m1) {
    // O(n)
    fp32 max = *std::max_element(m1.begin(), m1.end());
    fp32 sum = 0.0f;
    for(size_t i = 0; i< row; i++) {
        m1[i] = std::exp(m1[i] - max);
        sum += m1[i];
    }
    fp32 inv = 1.0f / sum;
    for(size_t i = 0; i< row; i++) m1[i]*=inv;
}