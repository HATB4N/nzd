#ifndef IOPTIMIZER_H
#define IOPTIMIZER_H

#include <vector>
#include <memory>
#include <Common/Types.h>

class DenseLayer;

class IOptimizer {
public:
    IOptimizer(std::vector<std::unique_ptr<DenseLayer>>& layers) : _layers(layers) {}
    virtual ~IOptimizer() = default;

    virtual void step() = 0;

protected:
    std::vector<std::unique_ptr<DenseLayer>>& _layers;
};

#endif // IOPTIMIZER_H
