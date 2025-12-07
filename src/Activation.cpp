#include "Activation.h"
#include <cassert>
#include <algorithm>
#include <cmath>
#include <vector>

void Act::softmax(Matrix_T<fp32> &m) {
    size_t num_classes = m.col();
    size_t batch_size = m.row();
    auto& m_data = m.data(View::NT);

    for(size_t i = 0; i < batch_size; ++i) {
        auto row_start = m_data.begin() + i * num_classes;
        auto row_end = row_start + num_classes;
        
        fp32 max = *std::max_element(row_start, row_end);
        fp32 sum = 0.0f;

        for(auto it = row_start; it < row_end; it++) {
            *it = std::exp(*it - max);
            sum += *it;
        }

        fp32 inv_sum = (sum == 0.0f) ? 0.0f : 1.0f / sum;
        for(auto it = row_start; it < row_end; it++) {
            *it *= inv_sum;
        }
    }
}

void Act::relu(Matrix_T<fp32> &m1) {
    auto& m_data = m1.data(View::NT);
    for(size_t i = 0; i < m1.size(); ++i) {
        m_data[i] = std::max(fp32(0.0f), m_data[i]);
    }
}

void Act::l_relu(Matrix_T<fp32> &m1) {
    for(size_t i = 0; i< m1.size(); ++i) {
        m1.data(View::T)[i] = std::max(0.01f*m1.data(View::T)[i], m1.data(View::T)[i]); // fix to variable(0.01f)
    }   
}

void ActDifr::difr_linear(Matrix_T<fp32> &m1) {
    for(size_t i = 0; i< m1.size(); ++i) {
        m1.data(View::T)[i] = (fp32)1.0f;
    }    
}

void ActDifr::difr_softmax(Matrix_T<fp32> &m1) {

}

void ActDifr::difr_relu(Matrix_T<fp32> &m1) {
    for(size_t i = 0; i< m1.size(); ++i) {
        m1.data(View::T)[i] = (fp32)(m1.data(View::T)[i]> 0);
    }
}

void ActDifr::difr_l_relu(Matrix_T<fp32> &m1) {
    for(size_t i = 0; i< m1.size(); ++i) {
        auto t = m1.data(View::T)[i]; 
        m1.data(View::T)[i] = (t> 0) ? 0.01f : 1.0f;
    }
}