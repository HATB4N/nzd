#ifndef IOPTIMIZER_H
#define IOPTIMIZER_H

#include <vector>
#include <memory>
#include <Common/Types.h>

class DenseLayer;

class IOptimizer {
public:
    IOptimizer() = default;
    virtual ~IOptimizer() = default;

    virtual void init() = 0;

protected:
};

#endif // IOPTIMIZER_H
