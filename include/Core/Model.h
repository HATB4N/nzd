#ifndef MODEL_H
#define MODEL_H

#include "Initializers/Initializer.h"
#include "Optimizers/Optimizer.h"
#include "Core/DenseLayer.h"
#include <string>
#include <vector>
#include <fstream>
#include <cstddef>

class Model {
public:
    Model(uint64_t input_dim,
          uint64_t batch_size,
          InitType init,
          OptType opt);

    void add(uint64_t dim, ActType act);

    Matrix_T<fp32> forward_batch(const Matrix_T<fp32> &x);
    Matrix_T<fp32> backward_batch(const Matrix_T<fp32> &y);

    int save_parms(); // @ ModelIO.cpp
    int load_parms(); // @ ModelIO.cpp

private:
    std::vector<std::unique_ptr<DenseLayer> > _layers;
    InitType _init;
    OptType _opt;

    int save_unit_parms(uint64_t index, std::ofstream &_fout); // @ ModelIO.cpp
    int load_unit_parms(std::ifstream &_fin); // @ ModelIO.cpp
    std::string base_dir = "data/parms.nzd"; // init시 초기화시켜
    uint64_t _batch_size;
    uint64_t _input_dim;
    uint64_t _last_dim;
};

#endif // MODEL_H
