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
    explicit Adam();
    void init() { return; }

private:

};

#endif // ADAM_H
