#ifndef IOPTIMIZER_H
#define IOPTIMIZER_H

#include <vector>
#include <memory>
#include <Common/Types.h>
#include <Common/Struct.h>

class DenseLayer;

class IOptimizer {
public:
    IOptimizer(Matrix_T<fp32>& W,
               Matrix_T<fp32>& b,
               Matrix_T<fp32>& dW,
               Matrix_T<fp32>& db) : _W(W), _b(b), 
                                     _dW(dW), _db(db) {}
    virtual ~IOptimizer() = default;

    virtual void init() = 0;
    virtual void step() = 0;

protected:
Matrix_T<fp32>& _W;
Matrix_T<fp32>& _b;
Matrix_T<fp32>& _dW;
Matrix_T<fp32>& _db;
};

#endif // IOPTIMIZER_H
