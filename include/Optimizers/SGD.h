#ifndef SGD_H
#define SGD_H

#include "Optimizers/IOptimizer.h"
#include "Common/Types.h"
#include <vector>
#include <memory>

class DenseLayer;

class SGD : public IOptimizer {
public:
    explicit SGD();
    void init() { return; }
private:

};

#endif
