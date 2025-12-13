#ifndef MODEL_H
#define MODEL_H

#include "Initializers/Initializer.h"
#include "Core/DenseLayer.h"
#include <string>
#include <vector>
#include <fstream>
#include <cstddef>

class Model {
public:
    Model(uint64_t input_dim,
          uint64_t batch_size,
          InitType init);;
    void add(uint64_t dim, ActFunc act);
    Matrix_T<fp32> forward_batch(const Matrix_T<fp32>& x);
    Matrix_T<fp32> backward_batch(const Matrix_T<fp32>& y);
    void apply_softmax_cross_entropy_backward(Matrix_T<fp32>& dx, const std::vector<uint8_t>& labels);
    int save_parms(); // @ ModelIO.cpp
    int load_parms(); // @ ModelIO.cpp

private:
    uint64_t _nol;
    // _nol: only count hidden layers (ignore static components)
    // 0: input layer
    // 1, 2, ..., _nol: hidden layer
    // _nol+1: output layer
    std::vector<std::unique_ptr<DenseLayer>> _layers;
    InitType _init;
    int save_unit_parms(uint64_t index, std::ofstream& _fout); // @ ModelIO.cpp
    int load_unit_parms(std::ifstream& _fin); // @ ModelIO.cpp
    std::string base_dir = "data/parms.nzd"; // init시 초기화시켜
    uint64_t _batch_size;
    uint64_t _input_dim;
    uint64_t _last_dim;
};

#endif // MODEL_H