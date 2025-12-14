#ifndef SGD_H
#define SGD_H

#include "Optimizers/IOptimizer.h"
#include "Common/Types.h"
#include <vector>
#include <memory>

class DenseLayer;

class SGD : public IOptimizer {
public:
    explicit SGD(Matrix_T<fp32>& W,
                 Matrix_T<fp32>& b,
                 Matrix_T<fp32>& dW,
                 Matrix_T<fp32>& db) : IOptimizer(W, b, dW, db) {}
    void init() { return; }
    void step();
private:

};

#endif // SGD_H
