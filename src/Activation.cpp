#include "Activation.h"
#include <cassert>
#include <algorithm>
#include <cmath>

void Act::softmax(Matrix_T<fp32> &m1_t) {
    size_t _batch = m1_t.col();
    for(size_t i = 0; i< m1_t.row(); ++i) {
        auto start = m1_t.data(View::T).begin() + i * _batch;
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

void Act::relu(Matrix_T<fp32> &m1) {
    for(size_t i = 0; i< m1.size(); ++i) {
        m1.data(View::T)[i] = std::max(fp32(0.0f), m1.data(View::T)[i]);
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