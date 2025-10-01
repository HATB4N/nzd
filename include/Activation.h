#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <stdfloat>
#include <vector>
#include <cstddef>

using fp16 = std::float16_t;
using fp32 = std::float32_t;

class Activation {
public:
    Activation(size_t m1_row, size_t m1_col, size_t batch);
    void softmax(std::vector<fp32> &m1);
    void sigmoid(std::vector<fp32> &m1);
    void silu(std::vector<fp32> &m1);
    void relu(std::vector<fp32> &m1);
    void l_relu(std::vector<fp32> &m1);

private:
    size_t _m1_row, _m1_col, _batch;

};



#endif