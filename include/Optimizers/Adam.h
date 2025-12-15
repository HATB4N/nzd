#ifndef ADAM_H
#define ADAM_H

#include "Optimizers/IOptimizer.h"
#include "Common/Types.h"
#include "Common/Struct.h"
#include <vector>
#include <memory>

class DenseLayer;

class Adam : public IOptimizer {
public:
    explicit Adam(Matrix_T<fp32> &W,
                  Matrix_T<fp32> &b,
                  Matrix_T<fp32> &dW,
                  Matrix_T<fp32> &db) : IOptimizer(W, b, dW, db) {
    }

    void init() { return; }
    void step();

private:
};

#endif // ADAM_H
