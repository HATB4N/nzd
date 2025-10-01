#ifndef DENSELAYER_H
#define DENSELAYER_H

#include "Matrix.h"
#include "Activation.h"
#include <memory>
#include <vector>

using fp16 = std::float16_t;
using fp32 = std::float32_t;

class DenseLayer {
public:
    DenseLayer(size_t row, size_t col, size_t batch, unsigned int threads);
    ~DenseLayer() = default;
    // wip
    void forward();
    void backward();
    void update();

private:
    std::unique_ptr<Matrix> _gemm;
    std::unique_ptr<Activation> _activator;
    fp16 random_float(float fmin, float fmax);
    size_t _r, _c, _b;
    unsigned int _t;
};

#endif