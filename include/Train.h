#ifndef TRAIN_H
#define TRAIN_H

#include "Model.h"
#include "Utils/Mnist.h"

class Train {
public:
    Train(uint64_t epochs,
          uint64_t bpe,
          uint64_t input_dim, 
          uint64_t output_dim,
          uint64_t hidden_dim);
    void train_epoch();
private:
    std::unique_ptr<Model> _model;
    uint64_t _total_data;
    uint64_t _epochs;
    uint64_t _batches_per_epoch;
    uint64_t _input_dim;
    const uint64_t _output_dim;
    const uint64_t _hidden_dim;
    std::unique_ptr<Mnist> _mnist;
};

#endif // TRAIN_H